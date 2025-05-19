#include"imeo_ModelEngine_yolov11_obb.hpp"

#include<fstream>
#include<memory>
#include <iomanip> 
#include <sstream>
#include <algorithm>
#include <numeric>

namespace rw
{
	namespace imeo
	{
		ModelEngine_Yolov11_obb::ModelEngine_Yolov11_obb(const std::string& modelPath)
		{
			init(modelPath);
		}

		ModelEngine_Yolov11_obb::~ModelEngine_Yolov11_obb()
		{
			output_tensors[0].release();
		}

		void ModelEngine_Yolov11_obb::preprocess(const cv::Mat& mat)
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

		void ModelEngine_Yolov11_obb::infer()
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
		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_obb::postProcess()
		{
			cpu_output_buffer = output_tensors[0].GetTensorMutableData<float>();
			std::vector<Detection> boxes;

			const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

			for (int i = 0; i < det_output.cols; ++i) {
				const  cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
				cv::Point class_id_point;
				double score;
				minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

				if (score > config.conf_threshold) {
					const float cx = det_output.at<float>(0, i);
					const float cy = det_output.at<float>(1, i);
					const float ow = det_output.at<float>(2, i);
					const float oh = det_output.at<float>(3, i);
					const float angle = det_output.at<float>(4 + num_classes, i);
					Detection box;
					box.angle = angle;
					box.c_x = cx;
					box.c_y = cy;
					box.width = ow;
					box.height = oh;
					box.conf = score;
					box.class_id = class_id_point.y;
					boxes.push_back(box);
				}
			}
			std::vector<Detection> nms_boxes = rotatedNMS(boxes, config.nms_threshold);

			auto result = convertDetectionToDetectionRectangleInfo(nms_boxes);

			return result;
		}

		void ModelEngine_Yolov11_obb::init(const std::string& engine_path)
		{
			env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "yolo");
			Ort::SessionOptions options;
			OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0);
			std::wstring path = stringToWString(engine_path);
			session = Ort::Session(env, path.c_str(), options);
			Ort::AllocatorWithDefaultOptions allocator;

			input_name = session.GetInputNameAllocated(0, allocator).get();
			output_name = session.GetOutputNameAllocated(0, allocator).get();

			input_node_names.push_back(input_name.c_str());
			output_node_names.push_back(output_name.c_str());

			auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
			auto output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
			input_h = input_shape[2];
			input_w = input_shape[3];
			detection_attribute_size = output_shape[1];
			num_detections = output_shape[2];
			num_classes = output_shape[1] - 5;

			cv::Mat zero_mat = cv::Mat::zeros(input_h, input_w, CV_8UC3);
			preprocess(zero_mat);
			for (int i = 0; i < 10; i++) {
				this->infer();
			}
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_obb::convertDetectionToDetectionRectangleInfo(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());

			std::vector<Detection> postDections;
			if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				postDections = convertWhenResize(detections);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				postDections = convertWhenLetterBox(detections);
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				postDections = convertWhenCentralCrop(detections);
			}
			else
			{
				postDections = convertWhenResize(detections);
			}

			for (const auto& item : postDections)
			{
				DetectionRectangleInfo resultItem;

				float cx = item.c_x;
				float cy = item.c_y;
				float w = item.width;
				float h = item.height;
				float angle_rad = item.angle;

				// 以中心为原点，未旋转时的四个角点
				float dx[4] = { -w / 2,  w / 2,  w / 2, -w / 2 };
				float dy[4] = { -h / 2, -h / 2,  h / 2,  h / 2 };

				std::pair<int, int> corners[4];
				for (int i = 0; i < 4; ++i) {
					float x_rot = dx[i] * std::cos(angle_rad) - dy[i] * std::sin(angle_rad);
					float y_rot = dx[i] * std::sin(angle_rad) + dy[i] * std::cos(angle_rad);
					corners[i].first = static_cast<int>(std::round(cx + x_rot));
					corners[i].second = static_cast<int>(std::round(cy + y_rot));
				}

				resultItem.leftTop = corners[0];
				resultItem.rightTop = corners[1];
				resultItem.rightBottom = corners[2];
				resultItem.leftBottom = corners[3];
				resultItem.center_x = static_cast<int>(std::round(cx));
				resultItem.center_y = static_cast<int>(std::round(cy));
				resultItem.width = static_cast<int>(std::round(w));
				resultItem.height = static_cast<int>(std::round(h));
				resultItem.area = static_cast<long>(std::round(w * h));
				resultItem.classId = static_cast<size_t>(item.class_id);
				resultItem.score = item.conf;

				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::convertWhenLetterBox(
			const std::vector<Detection>& detections)
		{
			std::vector<Detection> result;
			result.reserve(detections.size());
			const auto& params = centerCropParams;
			float scale = letterBoxScale;
			int dw = letterBoxdw;
			int dh = letterBoxdh;
			for (const auto& item : detections)
			{
				float x1 = (item.c_x - dw) / scale;
				float y1 = (item.c_y - dh) / scale;
				float x2 = (item.c_x + item.width - dw) / scale;
				float y2 = (item.c_y + item.height - dh) / scale;
				Detection resultItem;
				resultItem.c_x = x1;
				resultItem.c_y = y1;
				resultItem.width = x2 - x1;
				resultItem.height = y2 - y1;
				resultItem.angle = item.angle;
				resultItem.conf = item.conf;
				resultItem.class_id = item.class_id;
				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::convertWhenCentralCrop(
			const std::vector<Detection>& detections)
		{
			std::vector<Detection> result;
			result.reserve(detections.size());
			const auto& params = centerCropParams;
			for (const auto& item : detections)
			{
				float x1 = item.c_x + params.crop_x - params.pad_left;
				float y1 = item.c_y + params.crop_y - params.pad_top;
				float x2 = item.c_x + item.width + params.crop_x - params.pad_left;
				float y2 = item.c_y + item.height + params.crop_y - params.pad_top;

				Detection resultItem;
				resultItem.c_x = (x1 + x2) / 2;
				resultItem.c_y = (y1 + y2) / 2;
				resultItem.width = x2 - x1;
				resultItem.height = y2 - y1;
				resultItem.angle = item.angle;
				resultItem.conf = item.conf;
				resultItem.class_id = item.class_id;
				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::convertWhenResize(
			const std::vector<Detection>& detections)
		{
			std::vector<Detection> result;
			result.reserve(detections.size());
			auto scaleX = sourceWidth / static_cast<float>(input_w);
			auto scaleY = sourceHeight / static_cast<float>(input_h);
			for (const auto& item : detections)
			{
				Detection resultItem;
				resultItem.width = item.width * scaleX;
				resultItem.height = item.height * scaleY;
				resultItem.c_x = item.c_x * scaleX;
				resultItem.c_y = item.c_y * scaleY;
				resultItem.angle = item.angle;
				resultItem.conf = item.conf;
				resultItem.class_id = item.class_id;
				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<ModelEngine_Yolov11_obb::Detection> ModelEngine_Yolov11_obb::rotatedNMS(
			const std::vector<ModelEngine_Yolov11_obb::Detection>& dets, double iouThreshold)
		{
			std::vector<ModelEngine_Yolov11_obb::Detection> result;
			if (dets.empty()) return result;

			// 按置信度降序排序
			std::vector<size_t> idxs(dets.size());
			std::iota(idxs.begin(), idxs.end(), 0);
			std::sort(idxs.begin(), idxs.end(), [&](size_t i, size_t j) {
				return dets[i].conf > dets[j].conf;
				});

			std::vector<bool> suppressed(dets.size(), false);

			for (size_t i = 0; i < idxs.size(); ++i) {
				if (suppressed[idxs[i]]) continue;
				result.push_back(dets[idxs[i]]);
				for (size_t j = i + 1; j < idxs.size(); ++j) {
					if (suppressed[idxs[j]]) continue;
					if (rotatedIoU(dets[idxs[i]], dets[idxs[j]]) > iouThreshold) {
						suppressed[idxs[j]] = true;
					}
				}
			}
			return result;
		}

		double ModelEngine_Yolov11_obb::rotatedIoU(const ModelEngine_Yolov11_obb::Detection& a,
			const ModelEngine_Yolov11_obb::Detection& b)
		{
			cv::RotatedRect rect1 = toRotatedRect(a);
			cv::RotatedRect rect2 = toRotatedRect(b);

			std::vector<cv::Point2f> intersection;
			auto interType = cv::rotatedRectangleIntersection(rect1, rect2, intersection);

			if (interType == cv::INTERSECT_NONE || intersection.empty())
				return 0.0;

			double interArea = cv::contourArea(intersection);
			double area1 = rect1.size.area();
			double area2 = rect2.size.area();
			double iou = interArea / (area1 + area2 - interArea);
			return iou;
		}

		cv::RotatedRect ModelEngine_Yolov11_obb::toRotatedRect(const ModelEngine_Yolov11_obb::Detection& det)
		{
			return cv::RotatedRect(
				cv::Point2f(det.c_x + det.width / 2.0f, det.c_y + det.height / 2.0f), // center
				cv::Size2f(det.width, det.height),
				det.angle
			);
		}

		cv::Mat ModelEngine_Yolov11_obb::draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList)
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

		std::wstring ModelEngine_Yolov11_obb::stringToWString(const std::string& str)
		{
			size_t len = std::mbstowcs(nullptr, str.c_str(), 0);
			std::wstring wstr(len, L'\0');
			std::mbstowcs(&wstr[0], str.c_str(), len);
			return wstr;
		}
	}
}
