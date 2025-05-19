#include"imeo_ModelEngine_yolov11_seg.hpp"

#include <cuda_runtime_api.h>

namespace rw
{
	namespace imeo
	{
		ModelEngine_Yolov11_Seg::ModelEngine_Yolov11_Seg(const std::string& modelPath)
		{
			init(modelPath);
		}

		ModelEngine_Yolov11_Seg::~ModelEngine_Yolov11_Seg()
		{
			output_tensors[0].release();
		}

		void ModelEngine_Yolov11_Seg::preprocess(const cv::Mat& mat)
		{
			sourceWidth = mat.cols;
			sourceHeight = mat.rows;

			if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				infer_image = PreProcess::letterbox(mat, input_w, input_h, config.letterBoxColor, letterBoxScale, letterBoxdw, letterBoxdh);
				infer_image = cv::dnn::blobFromImage(infer_image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				infer_image = PreProcess::centerCrop(mat, input_w, input_h, config.centerCropColor, &centerCropParams);
				infer_image = cv::dnn::blobFromImage(infer_image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
			}
			else
			{
				 infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
			}

			std::vector<int64_t>input_node_dims = { 1,3,input_h,input_h };
			auto input_size = 1 * 3 * input_h * input_w;
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

		void ModelEngine_Yolov11_Seg::infer()
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
		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_Seg::postProcess()
		{
			std::vector<Detection> output;


			cpu_output_buffer = output_tensors[0].GetTensorMutableData<float>();
			std::vector<cv::Rect> boxes;
			std::vector<int> class_ids;
			std::vector<float> confidences;

			const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

			for (int i = 0; i < det_output.cols - maskCoefficientNum; ++i) {
				const cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
				cv::Point class_id_point;
				double score;
				cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

				if (score > config.conf_threshold) {
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
			cv::dnn::NMSBoxes(boxes, confidences, config.conf_threshold, config.nms_threshold, nms_result);

			for (int i = 0; i < nms_result.size(); i++)
			{
				Detection result;
				int idx = nms_result[i];
				result.class_id = class_ids[idx];
				result.conf = confidences[idx];
				result.bbox = boxes[idx];
				output.push_back(result);
			}

			return convertDetectionToDetectionRectangleInfo(output);
		}

		void ModelEngine_Yolov11_Seg::init(const std::string& engine_path)
		{
			env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "yolo");
			Ort::SessionOptions options;
			OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0);
			std::wstring path = stringToWString(engine_path);
			session = Ort::Session(env, path.c_str(), options);
			Ort::AllocatorWithDefaultOptions allocator;

			input_name = session.GetInputNameAllocated(0, allocator).get();
			output_name = session.GetOutputNameAllocated(0, allocator).get();
			output_name1 = session.GetOutputNameAllocated(1, allocator).get();

			for (int i = 0;i < session.GetInputCount();++i)
			{
				session.GetInputNameAllocated(i, allocator);
			}

			input_node_names.push_back(input_name.c_str());
			output_node_names.push_back(output_name.c_str());
			output_node_names.push_back(output_name1.c_str());

			auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
			auto output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
			input_h = input_shape[2];
			input_w = input_shape[3];
			detection_attribute_size = output_shape[1];
			num_detections = output_shape[2];
			maskCoefficientNum = 32;
			num_classes = output_shape[1] - 4- maskCoefficientNum;

			cv::Mat zero_mat = cv::Mat::zeros(input_h, input_w, CV_8UC3);
			preprocess(zero_mat);
			for (int i = 0; i < 10; i++) {
				this->infer();
			}
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_Seg::convertDetectionToDetectionRectangleInfo(
			const std::vector<Detection>& detections)
		{
			if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				return convertWhenResize(detections);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				return convertWhenLetterBox(detections);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				return convertWhenCentralCrop(detections);
			}
			else
			{
				return convertWhenResize(detections);
			}
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_Seg::convertWhenResize(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());
			auto scaleX = sourceWidth / static_cast<float>(input_w);
			auto scaleY = sourceHeight / static_cast<float>(input_h);
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

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_Seg::convertWhenLetterBox(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());

			// letterBoxScale: scale , letterBoxdw/letterBoxdh: padding
			float scale = letterBoxScale;
			int dw = letterBoxdw;
			int dh = letterBoxdh;

			for (const auto& item : detections)
			{
				DetectionRectangleInfo resultItem;

				// 反算到原图坐标
				float x1 = (item.bbox.x - dw) / scale;
				float y1 = (item.bbox.y - dh) / scale;
				float x2 = (item.bbox.x + item.bbox.width - dw) / scale;
				float y2 = (item.bbox.y + item.bbox.height - dh) / scale;

				resultItem.leftTop.first = x1;
				resultItem.leftTop.second = y1;
				resultItem.rightTop.first = x2;
				resultItem.rightTop.second = y1;
				resultItem.leftBottom.first = x1;
				resultItem.leftBottom.second = y2;
				resultItem.rightBottom.first = x2;
				resultItem.rightBottom.second = y2;
				resultItem.width = x2 - x1;
				resultItem.height = y2 - y1;
				resultItem.center_x = (x1 + x2) / 2;
				resultItem.center_y = (y1 + y2) / 2;
				resultItem.area = resultItem.width * resultItem.height;
				resultItem.classId = item.class_id;
				resultItem.score = item.conf;
				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_Seg::convertWhenCentralCrop(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());

			const auto& params = centerCropParams;

			for (const auto& item : detections)
			{
				//Calculate the original coordinates
				float x1 = item.bbox.x + params.crop_x - params.pad_left;
				float y1 = item.bbox.y + params.crop_y - params.pad_top;
				float x2 = item.bbox.x + item.bbox.width + params.crop_x - params.pad_left;
				float y2 = item.bbox.y + item.bbox.height + params.crop_y - params.pad_top;

				DetectionRectangleInfo resultItem;
				resultItem.leftTop.first = x1;
				resultItem.leftTop.second = y1;
				resultItem.rightTop.first = x2;
				resultItem.rightTop.second = y1;
				resultItem.leftBottom.first = x1;
				resultItem.leftBottom.second = y2;
				resultItem.rightBottom.first = x2;
				resultItem.rightBottom.second = y2;
				resultItem.width = x2 - x1;
				resultItem.height = y2 - y1;
				resultItem.center_x = (x1 + x2) / 2;
				resultItem.center_y = (y1 + y2) / 2;
				resultItem.area = resultItem.width * resultItem.height;
				resultItem.classId = item.class_id;
				resultItem.score = item.conf;
				result.push_back(resultItem);
			}
			return result;
		}

		cv::Mat ModelEngine_Yolov11_Seg::draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList)
		{
			cv::Mat result = mat.clone();
			ImagePainter::PainterConfig config;
			for (const auto& item : infoList)
			{
				std::ostringstream oss;
				oss << "classId:" << item.classId << " score:" << std::fixed << std::setprecision(2) << item.score;
				config.text = oss.str();
				ImagePainter::drawShapesOnSourceImg(result, item, config);
			}
			return result;
		}

		std::wstring ModelEngine_Yolov11_Seg::stringToWString(const std::string& str)
		{
			size_t len = std::mbstowcs(nullptr, str.c_str(), 0);
			std::wstring wstr(len, L'\0');
			std::mbstowcs(&wstr[0], str.c_str(), len);
			return wstr;
		}
	}
}
