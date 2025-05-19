#include"imet_ModelEngine_yolov11_obb.hpp"

#include"cuda_device_runtime_api.h"

#include<fstream>
#include<memory>
#include <iomanip> 
#include <sstream>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace rw
{
	namespace imet
	{

		ModelEngine_Yolov11_obb::ModelEngine_Yolov11_obb(const std::string& modelPath,
			nvinfer1::ILogger& logger)
		{
			init(modelPath, logger);
		}

		ModelEngine_Yolov11_obb::~ModelEngine_Yolov11_obb()
		{
			for (int i = 0; i < 2; i++)
				(cudaFree(gpu_buffers[i]));
			delete[] cpu_output_buffer;
			delete context;
			delete engine;
			delete runtime;
		}

		void ModelEngine_Yolov11_obb::init(std::string engine_path, nvinfer1::ILogger& logger)
		{
			std::ifstream engineStream(engine_path, std::ios::binary);
			engineStream.seekg(0, std::ios::end);
			const size_t modelSize = engineStream.tellg();
			engineStream.seekg(0, std::ios::beg);
			std::unique_ptr<char[]> engineData(new char[modelSize]);
			engineStream.read(engineData.get(), modelSize);
			engineStream.close();

			runtime = nvinfer1::createInferRuntime(logger);
			engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
			context = engine->createExecutionContext();

			input_h = engine->getTensorShape(engine->getIOTensorName(0)).d[2];
			input_w = engine->getTensorShape(engine->getIOTensorName(0)).d[3];
			detection_attribute_size = engine->getTensorShape(engine->getIOTensorName(1)).d[1];
			num_detections = engine->getTensorShape(engine->getIOTensorName(1)).d[2];
			num_classes = detection_attribute_size - 5;

			cpu_output_buffer = new float[num_detections * detection_attribute_size];
			(cudaMalloc((void**)&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));

			(cudaMalloc((void**)&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

			for (int i = 0;i < 10;i++) {
				this->infer();
			}
			cudaDeviceSynchronize();
		}

		void ModelEngine_Yolov11_obb::infer()
		{
			this->context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
			this->context->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
			this->context->enqueueV3(NULL);
		}

		

		
		

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_obb::postProcess()
		{
			std::vector<Detection> output;
			(cudaMemcpy(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
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
					const float angle= det_output.at<float>(4 + num_classes, i);
					Detection box;
                    box.angle = angle;
                    box.x = cx - ow / 2;
                    box.y = cy - oh / 2;
                    box.width = ow;
                    box.height = oh;
                    box.conf = score;
                    box.class_id = class_id_point.y;
					boxes.push_back(box);
				}
			}
			std::vector<Detection> nms_boxes = rotatedNMS(boxes, config.nms_threshold); // 0.1为IoU阈值，可根据实际调整

			auto result = convertDetectionToDetectionRectangleInfo(nms_boxes);

			return result;
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

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_obb::convertDetectionToDetectionRectangleInfo(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			auto scaleX = sourceWidth / static_cast<float>(input_w);
			auto scaleY = sourceHeight / static_cast<float>(input_h);
			result.reserve(detections.size());
			for (const auto& item : detections)
			{
				DetectionRectangleInfo resultItem;

				// 还原到原图坐标
				float cx = (item.x + item.width / 2.0f) * scaleX;
				float cy = (item.y + item.height / 2.0f) * scaleY;
				float w = item.width * scaleX;
				float h = item.height * scaleY;
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

		void ModelEngine_Yolov11_obb::preprocess(const cv::Mat& mat)
		{
			sourceWidth = mat.cols;
			sourceHeight = mat.rows;
			auto infer_image =
				cv::dnn::blobFromImage(mat,
					1.f / 255.f,
					cv::Size(input_w, input_h),
					cv::Scalar(0, 0, 0), true);//1、缩放cv::resize;2、系数变换；3、色域变换bgr->rgb；4、图像裁剪cv::crop;5、数据标准化(x-mean)/var

			(cudaMemcpy(gpu_buffers[0],
				infer_image.data,
				input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
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
				cv::Point2f(det.x + det.width / 2.0f, det.y + det.height / 2.0f), // center
				cv::Size2f(det.width, det.height),
				det.angle
			);
		}
	}
}
