#include "imet_ModelEngine_yolov11_seg_refacotr.hpp"

#include"cuda_device_runtime_api.h"

#include<fstream>
#include<memory>

namespace rw {
	namespace imet {
		void ModelEngine_Yolov11_seg_refactor::preprocess(const cv::Mat& mat)
		{
			_sourceWidth = mat.cols;
			_sourceHeight = mat.rows;

			if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				auto infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				cv::Mat letterbox_image = PreProcess::letterbox(mat, input_w, input_h, config.letterBoxColor, _letterBoxScale, _letterBoxdw, _letterBoxdh);
				auto infer_image = cv::dnn::blobFromImage(letterbox_image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else if (config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				cv::Mat center_crop_image = PreProcess::centerCrop(mat, input_w, input_h, config.centerCropColor, &_centerCropParams);
				auto infer_image = cv::dnn::blobFromImage(center_crop_image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else
			{
				auto infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
		}

		void ModelEngine_Yolov11_seg_refactor::infer()
		{
			this->context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
			this->context->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
			this->context->setOutputTensorAddress(engine->getIOTensorName(2), gpu_buffers[2]);
			this->context->enqueueV3(NULL);
		}
		cv::Mat ModelEngine_Yolov11_seg_refactor::postProcessAndDraw(cv::Mat& mat)
		{
			std::vector<DetectionSeg> output;
			(cudaMemcpy(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
			(cudaMemcpy(cpu_output_buffer2, gpu_buffers[2], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
			std::vector<cv::Rect> boxes;
			std::vector<int> class_ids;
			std::vector<float> confidences;
			std::vector<cv::Mat> mask_sigmoids;

			const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
			cv::Mat mask_protos(maskCoefficientNum, mask_h * mask_w, CV_32F, cpu_output_buffer2);

			for (int i = 0; i < det_output.cols; ++i) {
				const cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
				cv::Point class_id_point;
				double score;
				minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

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

					const cv::Mat mask_coeffs = det_output.col(i).rowRange(4 + num_classes, detection_attribute_size);
					cv::Mat mask_flat = mask_coeffs.t() * mask_protos; // [1, mask_h * mask_w]
					cv::Mat mask = mask_flat.reshape(1, mask_h); // [mask_h, mask_w]
					cv::Mat mask_sigmoid;
					cv::exp(-mask, mask_sigmoid);
					mask_sigmoid = 1.0 / (1.0 + mask_sigmoid);
					mask_sigmoids.push_back(mask_sigmoid);
				}
			}

			std::vector<int> nms_result = nmsWithKeepClass(
				boxes, class_ids, confidences, config.conf_threshold, config.nms_threshold, config.classids_nms_together);

			for (int i = 0; i < nms_result.size(); i++) {
				DetectionSeg result;
				int idx = nms_result[i];
				result.class_id = class_ids[idx];
				result.conf = confidences[idx];
				result.bbox = boxes[idx];
				result.mask_sigmoid = mask_sigmoids[idx];
				output.push_back(result);

				// 绘制 mask 到图片上
				cv::Mat mask_resized;
				cv::resize(result.mask_sigmoid, mask_resized, cv::Size(result.bbox.width, result.bbox.height), 0, 0, cv::INTER_LINEAR);
				cv::Mat mask_bin;
				cv::threshold(mask_resized, mask_bin, 0.5, 1.0, cv::THRESH_BINARY);

				// 取出 bbox 区域
				cv::Rect bbox = result.bbox;
				// 检查 bbox 是否在图像范围内
				cv::Rect img_rect(0, 0, mat.cols, mat.rows);
				cv::Rect roi = bbox & img_rect;
				if (roi.width <= 0 || roi.height <= 0) continue;

				// 只取有效区域
				cv::Mat mask_roi = mask_bin(cv::Rect(0, 0, roi.width, roi.height));
				cv::Mat img_roi = mat(roi);

				// 着色（以红色为例）
				std::vector<cv::Mat> channels;
				cv::split(img_roi, channels);
				// 红色通道加高亮
				channels[2].setTo(255, mask_roi > 0); // BGR: 2为R通道
				cv::merge(channels, img_roi);
			}

			masks = output;
			return mat;
		}
		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_seg_refactor::postProcess()
		{
			std::vector<DetectionSeg> output;
			(cudaMemcpy(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
			(cudaMemcpy(cpu_output_buffer2, gpu_buffers[2], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
			std::vector<cv::Rect> boxes;
			std::vector<int> class_ids;
			std::vector<float> confidences;
			std::vector<cv::Mat> mask_sigmoids;

			const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
			cv::Mat mask_protos(maskCoefficientNum, mask_h * mask_w, CV_32F, cpu_output_buffer2);

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
					cv::Rect box;
					box.x = static_cast<int>((cx - 0.5 * ow));
					box.y = static_cast<int>((cy - 0.5 * oh));
					box.width = static_cast<int>(ow);
					box.height = static_cast<int>(oh);

					boxes.push_back(box);
					class_ids.push_back(class_id_point.y);
					confidences.push_back(score);

					const  cv::Mat mask_coeffs = det_output.col(i).rowRange(4 + num_classes, detection_attribute_size);
					cv::Mat mask_flat = mask_coeffs.t() * mask_protos; // [1, mask_h * mask_w]
					cv::Mat mask = mask_flat.reshape(1, mask_h); // [mask_h, mask_w]
					cv::Mat mask_sigmoid;
					cv::exp(-mask, mask_sigmoid);
					mask_sigmoid = 1.0 / (1.0 + mask_sigmoid);
					mask_sigmoids.push_back(mask_sigmoid);
				}
			}

			std::vector<int> nms_result = nmsWithKeepClass(
				boxes, class_ids, confidences, config.conf_threshold, config.nms_threshold, config.classids_nms_together);

			for (int i = 0; i < nms_result.size(); i++)
			{
				DetectionSeg result;
				int idx = nms_result[i];
				result.class_id = class_ids[idx];
				result.conf = confidences[idx];
				result.bbox = boxes[idx];
				result.mask_sigmoid = mask_sigmoids[idx];
				output.push_back(result);
			}

			masks = output;
			auto result = convertToDetectionRectangleInfo(output);

			return result;
		}

		void ModelEngine_Yolov11_seg_refactor::init(const std::string& enginePath, nvinfer1::ILogger& logger)
		{
			std::ifstream engineStream(enginePath, std::ios::binary);
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
			maskCoefficientNum = engine->getTensorShape(engine->getIOTensorName(2)).d[1];
			mask_h = engine->getTensorShape(engine->getIOTensorName(2)).d[2];
			mask_w = engine->getTensorShape(engine->getIOTensorName(2)).d[3];
			detection_attribute_size = engine->getTensorShape(engine->getIOTensorName(1)).d[1];
			num_detections = engine->getTensorShape(engine->getIOTensorName(1)).d[2];
			num_classes = detection_attribute_size - 4 - maskCoefficientNum;

			auto thirdOutputSize = engine->getTensorShape(engine->getIOTensorName(2)).d[1];
			auto thirdOutputSize2 = engine->getTensorShape(engine->getIOTensorName(2)).d[2];
			auto thirdOutputSize3 = engine->getTensorShape(engine->getIOTensorName(2)).d[3];

			cpu_output_buffer = new float[num_detections * detection_attribute_size];
			(cudaMalloc((void**)&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));

			(cudaMalloc((void**)&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

			cpu_output_buffer2 = new float[thirdOutputSize * thirdOutputSize2 * thirdOutputSize3];
			(cudaMalloc((void**)&gpu_buffers[2], thirdOutputSize * thirdOutputSize2 * thirdOutputSize3 * sizeof(float)));

			for (int i = 0; i < 10; i++) {
				this->infer();
			}
			cudaDeviceSynchronize();
		}

		ModelEngine_Yolov11_seg_refactor::ModelEngine_Yolov11_seg_refactor(const std::string& modelPath,
			nvinfer1::ILogger& logger)
		{
			init(modelPath, logger);
		}

		ModelEngine_Yolov11_seg_refactor::~ModelEngine_Yolov11_seg_refactor()
		{
			for (int i = 0; i < 2; i++)
				(cudaFree(gpu_buffers[i]));
			delete[] cpu_output_buffer;
			delete[] cpu_output_buffer2;
			delete context;
			delete engine;
			delete runtime;
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_seg_refactor::convertToDetectionRectangleInfo(const std::vector<DetectionSeg>& detections)
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

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_seg_refactor::convertWhenResize(
			const std::vector<DetectionSeg>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());
			auto scaleX = _sourceWidth / static_cast<float>(input_w);
			auto scaleY = _sourceHeight / static_cast<float>(input_h);
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

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_seg_refactor::convertWhenCentralCrop(
			const std::vector<DetectionSeg>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());

			const auto& params = _centerCropParams;

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

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_seg_refactor::convertWhenLetterBox(
			const std::vector<DetectionSeg>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());

			float scale = _letterBoxScale;
			int dw = _letterBoxdw;
			int dh = _letterBoxdh;

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

		cv::Mat ModelEngine_Yolov11_seg_refactor::draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList)
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
			for (int i = 0; i < masks.size(); i++)
			{
				// 计算比例因子
				float scaleX = _sourceWidth / static_cast<float>(input_w);
				float scaleY = _sourceHeight / static_cast<float>(input_h);

				// 调整 bbox 到原图比例
				cv::Rect bbox = masks[i].bbox;
				bbox.x = static_cast<int>(bbox.x * scaleX);
				bbox.y = static_cast<int>(bbox.y * scaleY);
				bbox.width = static_cast<int>(bbox.width * scaleX);
				bbox.height = static_cast<int>(bbox.height * scaleY);

				// 检查 bbox 是否在图像范围内
				cv::Rect img_rect(0, 0, result.cols, result.rows);
				cv::Rect roi = bbox & img_rect;
				if (roi.width <= 0 || roi.height <= 0) continue;

				// 调整 mask 到 bbox 尺寸
				cv::Mat mask_resized;
				cv::resize(masks[i].mask_sigmoid, mask_resized, cv::Size(bbox.width, bbox.height), 0, 0, cv::INTER_LINEAR);

				// 二值化 mask
				cv::Mat mask_bin;
				cv::threshold(mask_resized, mask_bin, 0.5, 1.0, cv::THRESH_BINARY);

				// 只取有效区域
				cv::Mat mask_roi = mask_bin(cv::Rect(0, 0, roi.width, roi.height));
				cv::Mat img_roi = result(roi);

				// 着色（以红色为例）
				std::vector<cv::Mat> channels;
				cv::split(img_roi, channels);
				channels[2].setTo(255, mask_roi > 0); // BGR: 2为R通道
				cv::merge(channels, img_roi);
			}


			//postProcessAndDraw(result);
			return result;
		}
	}
}