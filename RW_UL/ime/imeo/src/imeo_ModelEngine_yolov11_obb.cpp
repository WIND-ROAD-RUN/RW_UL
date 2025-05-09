#include"imeo_ModelEngine_yolov11_obb.hpp"

namespace rw
{
	namespace imeo
	{
		ModelEngine_yolov11_obb::ModelEngine_yolov11_obb(const std::string& modelPath)
		{
			init(modelPath);
		}

		ModelEngine_yolov11_obb::~ModelEngine_yolov11_obb()
		{
			output_tensors[0].release();
		}

		void ModelEngine_yolov11_obb::preprocess(const cv::Mat& mat)
		{
			sourceWidth = mat.cols;
			sourceHeight = mat.rows;

			infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);

			std::vector<int64_t>input_node_dims = { 1,3,640,640 };
			auto input_size = 1 * 3 * 640 * 640;
			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
			input_tensor = Ort::Value::CreateTensor(
				memory_info,
				(float*)infer_image.data,
				input_size,
				input_node_dims.data(),
				input_node_dims.size()
			);
			ort_inputs.clear();
			ort_inputs.push_back(std::move(input_tensor));
		}

		void ModelEngine_yolov11_obb::infer()
		{

			output_tensors = session.Run(
				Ort::RunOptions{ nullptr },
				(const char* const*)input_node_names.data(),
				ort_inputs.data(),
				ort_inputs.size(),
				(const char* const*)output_node_names.data(),
				output_node_names.size()
			);
		}
		std::vector<DetectionRectangleInfo> ModelEngine_yolov11_obb::postProcess()
		{
			std::vector<Detection> output;

			cpu_output_buffer = output_tensors[0].GetTensorMutableData<float>();

			std::vector<cv::Rect> boxes;
			std::vector<int> class_ids;
			std::vector<float> confidences;

			const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

			for (int i = 0; i < det_output.cols; ++i) {
				const cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
				cv::Point class_id_point;
				double score;
				cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

				if (score > conf_threshold) {
					const float cx = det_output.at<float>(0, i);
					const float cy = det_output.at<float>(1, i);
					const float ow = det_output.at<float>(2, i);
					const float oh = det_output.at<float>(3, i);
					cv::Rect box;
					box.x = static_cast<int>((cx - 0.5 * ow));
					box.y = static_cast<int>((cy - 0.5 * oh));
					box.width = static_cast<int>(ow);
					box.height = static_cast<int>(oh);

					boxes.push_back(box);
					class_ids.push_back(class_id_point.y);
					confidences.push_back(score);
				}
			}

			std::vector<int> nms_result;
			cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

			for (int i = 0; i < nms_result.size(); i++)
			{
				Detection result;
				int idx = nms_result[i];
				result.class_id = class_ids[idx];
				result.conf = confidences[idx];
				result.bbox = boxes[idx];
				output.push_back(result);
			}


		}

		void ModelEngine_yolov11_obb::init(const std::string& engine_path)
		{
			std::cout << "onnxruntime infer" << std::endl;
			//onnxruntime
			env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "yolo");
			Ort::SessionOptions options;
			OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0);
			//OrtCUDAProviderOptions cudaOptions;
			//options.AppendExecutionProvider_CUDA(cudaOptions);
			//const wchar_t* path = L"yolo11n.onnx";
			std::wstring path = LR"(D:\Workplace\rep\RW_UL\Project\yolo\build\yolo11n.onnx)";
			session = Ort::Session(env, path.c_str(), options);
			Ort::AllocatorWithDefaultOptions allocator;
			//for (int i = 0;i < session.GetInputCount();++i)
			//{
			//	session.GetInputNameAllocated(i, allocator)
			//}
			auto input_name = session.GetInputNameAllocated(0, allocator);
			auto output_name = session.GetOutputNameAllocated(0, allocator);
			input_node_names.push_back("images");
			output_node_names.push_back("output0");
			auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
			auto output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
			input_h = input_shape[2];
			input_w = input_shape[3];
			detection_attribute_size = output_shape[1];
			num_detections = output_shape[2];
			num_classes = output_shape[1] - 4;
			if (true) {
				cv::Mat zero_mat = cv::Mat::zeros(input_h, input_w, CV_32FC3);
				preprocess(zero_mat);
				for (int i = 0; i < 10; i++) {
					this->infer();
				}
				printf("model warmup 10 times\n");
			}

		}

		std::vector<DetectionRectangleInfo> ModelEngine_yolov11_obb::convertDetectionToDetectionRectangleInfo(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			auto scaleX = sourceWidth / static_cast<float>(input_w);
			auto scaleY = sourceHeight / static_cast<float>(input_h);
			result.reserve(detections.size());
			for (const auto& item : detections)
			{
				DetectionRectangleInfo resultItem;
				resultItem.width = item.bbox.width * scaleX;
				resultItem.height = item.bbox.height * scaleY;
				resultItem.leftTop.first = item.bbox.x * scaleX;
				resultItem.leftTop.second = item.bbox.y * scaleY;
				resultItem.rightTop.first = item.bbox.x * scaleX + item.bbox.width * scaleX;
				resultItem.rightTop.second = item.bbox.y * scaleY;
				resultItem.leftBottom.first = item.bbox.x * scaleX;
				resultItem.leftBottom.second = item.bbox.y * scaleY + item.bbox.height * scaleY;
				resultItem.rightBottom.first = item.bbox.x * scaleX + item.bbox.width * scaleX;
				resultItem.rightBottom.second = item.bbox.y * scaleY + item.bbox.height * scaleY;
				resultItem.center_x = item.bbox.x * scaleX + item.bbox.width * scaleX / 2;
				resultItem.center_y = item.bbox.y * scaleY + item.bbox.height * scaleY / 2;
				resultItem.area = item.bbox.width * scaleX * item.bbox.height * scaleY;
				resultItem.classId = item.class_id;
				resultItem.score = item.conf;
				result.push_back(resultItem);
			}
			return result;
		}
	}
}
