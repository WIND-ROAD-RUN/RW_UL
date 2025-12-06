#include"imet_ModelEngine_yolov11_det.hpp"

#include <cuda_runtime.h>

#include<fstream>
#include<memory>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace rw
{
	namespace imet
	{
		ModelEngine_Yolov11_det::ModelEngine_Yolov11_det(const std::string& modelPath,
			nvinfer1::ILogger& logger)
		{
			init(modelPath, logger);
		}

		ModelEngine_Yolov11_det::~ModelEngine_Yolov11_det()
		{
			for (int i = 0; i < 2; i++)
				(cudaFree(_gpu_buffers[i]));
			delete[] _cpu_output_buffer;
			delete _context;
			delete _engine;
			delete _runtime;
		}

		void ModelEngine_Yolov11_det::init(std::string enginePath, nvinfer1::ILogger& logger)
		{
			std::ifstream engineStream(enginePath, std::ios::binary);
			engineStream.seekg(0, std::ios::end);
			const size_t modelSize = engineStream.tellg();
			engineStream.seekg(0, std::ios::beg);
			std::unique_ptr<char[]> engineData(new char[modelSize]);
			engineStream.read(engineData.get(), modelSize);
			engineStream.close();

			_runtime = nvinfer1::createInferRuntime(logger);
			_engine = _runtime->deserializeCudaEngine(engineData.get(), modelSize);
			_context = _engine->createExecutionContext();

			// ========== 添加调试代码：打印所有张量信息 ==========
			int nbTensors = _engine->getNbIOTensors();
			std::cout << "========== TensorRT 模型张量信息 (det) ==========" << std::endl;
			std::cout << "总张量数: " << nbTensors << std::endl;
			for (int i = 0; i < nbTensors; i++) {
				const char* name = _engine->getIOTensorName(i);
				auto dims = _engine->getTensorShape(name);
				auto ioMode = _engine->getTensorIOMode(name);

				std::cout << "张量[" << i << "]: " << name
					<< " | " << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
					<< " | Shape: [";
				for (int j = 0; j < dims.nbDims; j++) {
					std::cout << dims.d[j];
					if (j < dims.nbDims - 1) std::cout << ", ";
				}
				std::cout << "]" << std::endl;
			}
			std::cout << "===========================================" << std::endl;

			// 读取输入尺寸
			_input_h = _engine->getTensorShape(_engine->getIOTensorName(0)).d[2];
			_input_w = _engine->getTensorShape(_engine->getIOTensorName(0)).d[3];

			// ========== 根据张量数量判断模型类型 ==========
			if (nbTensors == 2) {
				// YOLOv11 标准检测模型：只有输入和检测输出
				// 张量[1]: 检测输出 [1, detection_attribute_size, num_detections]
				auto output_dims = _engine->getTensorShape(_engine->getIOTensorName(1));
				_detection_attribute_size = output_dims.d[1];
				_num_detections = output_dims.d[2];
				_num_classes = _detection_attribute_size - 4;
			}
			else if (nbTensors == 3) {
				// YOLOv11 分割模型：有输入、掩膜原型和检测输出
				// 张量[1]: 掩膜原型 [1, 32, mask_h, mask_w]
				// 张量[2]: 检测输出 [1, detection_attribute_size, num_detections]
				auto det_dims = _engine->getTensorShape(_engine->getIOTensorName(2));
				_detection_attribute_size = det_dims.d[1];
				_num_detections = det_dims.d[2];
				_num_classes = _detection_attribute_size - 4 - 32;  // 减去4个bbox + 32个mask系数
			}
			else if (nbTensors == 5) {
				// 旧版 YOLO 格式（YOLOv5/v7/v8早期）：多尺度输出 + 合并输出
				// 张量[1-3]: 多尺度输出 [1, 3, H, W, detection_attribute_size]
				// 张量[4]: 合并输出 [1, num_detections, detection_attribute_size]
				auto output_dims = _engine->getTensorShape(_engine->getIOTensorName(4));
				
				// 判断输出格式：[batch, num_detections, attrs] 或 [batch, attrs, num_detections]
				if (output_dims.nbDims == 3) {
					int dim1 = output_dims.d[1];
					int dim2 = output_dims.d[2];
					
					// 通常 num_detections 会远大于 detection_attribute_size
					if (dim1 > dim2) {
						// [1, num_detections, detection_attribute_size] 格式
						_num_detections = dim1;
						_detection_attribute_size = dim2;
					} else {
						// [1, detection_attribute_size, num_detections] 格式
						_detection_attribute_size = dim1;
						_num_detections = dim2;
					}
				}
				_num_classes = _detection_attribute_size - 4;
			}
			else {
				// 尝试自动查找输出张量
				std::cout << "警告：未知的模型结构（张量数=" << nbTensors << "），尝试自动查找输出张量..." << std::endl;
				
				bool found = false;
				for (int i = nbTensors - 1; i >= 0; i--) {
					auto ioMode = _engine->getTensorIOMode(_engine->getIOTensorName(i));
					if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
						auto dims = _engine->getTensorShape(_engine->getIOTensorName(i));
						
						// 查找类似 [batch, num_detections, attrs] 或 [batch, attrs, num_detections] 的输出
						if (dims.nbDims == 3) {
							int dim1 = dims.d[1];
							int dim2 = dims.d[2];
							
							if (dim1 > dim2) {
								_num_detections = dim1;
								_detection_attribute_size = dim2;
							} else {
								_detection_attribute_size = dim1;
								_num_detections = dim2;
							}
							
							_num_classes = _detection_attribute_size - 4;
							if (_num_classes > 0) {
								std::cout << "找到输出张量[" << i << "]: " << _engine->getIOTensorName(i) << std::endl;
								found = true;
								break;
							}
						}
					}
				}
				
				if (!found) {
					throw std::runtime_error("无法识别的模型结构：张量数量 = " + std::to_string(nbTensors));
				}
			}

			// ========== 参数验证 ==========
			std::cout << "========== 模型参数验证 (det) ==========" << std::endl;
			std::cout << "input_w: " << _input_w << ", input_h: " << _input_h << std::endl;
			std::cout << "detection_attribute_size: " << _detection_attribute_size << std::endl;
			std::cout << "num_detections: " << _num_detections << std::endl;
			std::cout << "num_classes: " << _num_classes << std::endl;

			if (_num_classes <= 0) {
				throw std::runtime_error("计算得到的 num_classes <= 0，模型结构可能不正确！");
			}
			std::cout << "===================================" << std::endl;

			// 分配内存
			_cpu_output_buffer = new float[_num_detections * _detection_attribute_size];
			(cudaMalloc((void**)&_gpu_buffers[0], 3 * _input_w * _input_h * sizeof(float)));
			(cudaMalloc((void**)&_gpu_buffers[1], _detection_attribute_size * _num_detections * sizeof(float)));

			// 预热
			for (int i = 0; i < 10; i++) {
				this->infer();
			}
			cudaDeviceSynchronize();
		}

		void ModelEngine_Yolov11_det::infer()
		{
			// 获取张量数量
			int nbTensors = _engine->getNbIOTensors();

			if (nbTensors == 2) {
				// YOLOv11 标准检测模型
				const int inputIndex = 0;
				const int outputIndex = 1;

				void* bindings[2];
				bindings[inputIndex] = _gpu_buffers[0];
				bindings[outputIndex] = _gpu_buffers[1];

				_context->executeV2(bindings);
			}
			else if (nbTensors == 3) {
				// YOLOv11 分割模型（但 det 类只使用检测输出）
				// 0: images (INPUT)
				// 1: output1 (掩膜原型) - 不使用
				// 2: output0 (检测结果)
				const int inputIndex = 0;
				const int outputDetIndex = 2;  // 检测输出在索引2

				void* bindings[3];
				bindings[inputIndex] = _gpu_buffers[0];
				bindings[1] = nullptr;  // 掩膜原型不需要
				bindings[outputDetIndex] = _gpu_buffers[1];

				_context->executeV2(bindings);
			}
			else if (nbTensors == 5) {
				// 旧版 YOLO 格式：使用最后一个输出（合并输出）
				const int inputIndex = 0;
				const int outputIndex = 4;  // 使用合并后的输出

				void* bindings[5];
				bindings[inputIndex] = _gpu_buffers[0];
				bindings[1] = nullptr;  // 多尺度输出1 - 不使用
				bindings[2] = nullptr;  // 多尺度输出2 - 不使用
				bindings[3] = nullptr;  // 多尺度输出3 - 不使用
				bindings[outputIndex] = _gpu_buffers[1];

				_context->executeV2(bindings);
			}
			else {
				// 通用处理：使用最后一个输出张量
				std::vector<void*> bindings(nbTensors, nullptr);
				bindings[0] = _gpu_buffers[0];  // 输入
				bindings[nbTensors - 1] = _gpu_buffers[1];  // 最后一个输出

				_context->executeV2(bindings.data());
			}
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_det::postProcess()
		{
			std::vector<Detection> output;
			(cudaMemcpy(_cpu_output_buffer, _gpu_buffers[1], _num_detections * _detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
			std::vector<cv::Rect> boxes;
			std::vector<int> class_ids;
			std::vector<float> confidences;

			// 判断数据布局：[num_detections, detection_attribute_size] 或 [detection_attribute_size, num_detections]
			// 根据 _num_detections 和 _detection_attribute_size 的大小关系判断
			cv::Mat det_output;
			if (_num_detections > _detection_attribute_size) {
				// 数据是 [num_detections, detection_attribute_size] 格式，需要转置
				det_output = cv::Mat(_num_detections, _detection_attribute_size, CV_32F, _cpu_output_buffer);
				det_output = det_output.t();  // 转置为 [detection_attribute_size, num_detections]
			} else {
				// 数据已经是 [detection_attribute_size, num_detections] 格式
				det_output = cv::Mat(_detection_attribute_size, _num_detections, CV_32F, _cpu_output_buffer);
			}

			for (int i = 0; i < det_output.cols; ++i) {
				const cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + _num_classes);
				cv::Point class_id_point;
				double score;
				minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

				if (score > _config.conf_threshold) {
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

			std::vector<int> nms_result = nmsWithKeepClass(
				boxes, class_ids, confidences, _config.conf_threshold, _config.nms_threshold, _config.classids_nms_together);

			for (int i = 0; i < nms_result.size(); i++)
			{
				Detection result;
				int idx = nms_result[i];
				result.class_id = class_ids[idx];
				result.conf = confidences[idx];
				result.rect = boxes[idx];
				output.push_back(result);
			}
			auto size = output.size();
			if (size == 0) {
				return {};
			}

			auto result = convertDetectionToDetectionRectangleInfo(output);

			return result;
		}

		cv::Mat ModelEngine_Yolov11_det::draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList)
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

		void ModelEngine_Yolov11_det::preprocess(const cv::Mat& mat)
		{
			_sourceWidth = mat.cols;
			_sourceHeight = mat.rows;

			if (_config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				auto infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(_input_w, _input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(_gpu_buffers[0], infer_image.data, _input_w * _input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else if (_config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				cv::Mat letterbox_image = PreProcess::letterbox(mat, _input_w, _input_h, _config.letterBoxColor, letterBoxScale, letterBoxdw, letterBoxdh);
				auto infer_image = cv::dnn::blobFromImage(letterbox_image, 1.f / 255.f, cv::Size(_input_w, _input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(_gpu_buffers[0], infer_image.data, _input_w * _input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else if (_config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				cv::Mat center_crop_image = PreProcess::centerCrop(mat, _input_w, _input_h, _config.centerCropColor, &_centerCropParams);
				auto infer_image = cv::dnn::blobFromImage(center_crop_image, 1.f / 255.f, cv::Size(_input_w, _input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(_gpu_buffers[0], infer_image.data, _input_w * _input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
			else
			{
				auto infer_image = cv::dnn::blobFromImage(mat, 1.f / 255.f, cv::Size(_input_w, _input_h), cv::Scalar(0, 0, 0), true);
				(cudaMemcpy(_gpu_buffers[0], infer_image.data, _input_w * _input_h * mat.channels() * sizeof(float), cudaMemcpyHostToDevice));
			}
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_det::convertDetectionToDetectionRectangleInfo(
			const std::vector<Detection>& detections)
		{
			if (_config.imagePretreatmentPolicy == ImagePretreatmentPolicy::Resize)
			{
				return convertWhenResize(detections);
			}
			else if (_config.imagePretreatmentPolicy == ImagePretreatmentPolicy::LetterBox)
			{
				return convertWhenLetterBox(detections);
			}
			else if (_config.imagePretreatmentPolicy == ImagePretreatmentPolicy::CenterCrop)
			{
				return convertWhenCentralCrop(detections);
			}
			else
			{
				return convertWhenResize(detections);
			}
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_det::convertWhenResize(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());
			auto scaleX = _sourceWidth / static_cast<float>(_input_w);
			auto scaleY = _sourceHeight / static_cast<float>(_input_h);
			for (const auto& item : detections)
			{
				DetectionRectangleInfo resultItem;
				resultItem.width = item.rect.width * scaleX;
				resultItem.height = item.rect.height * scaleY;
				resultItem.leftTop.first = item.rect.x * scaleX;
				resultItem.leftTop.second = item.rect.y * scaleY;
				resultItem.rightTop.first = item.rect.x * scaleX + item.rect.width * scaleX;
				resultItem.rightTop.second = item.rect.y * scaleY;
				resultItem.leftBottom.first = item.rect.x * scaleX;
				resultItem.leftBottom.second = item.rect.y * scaleY + item.rect.height * scaleY;
				resultItem.rightBottom.first = item.rect.x * scaleX + item.rect.width * scaleX;
				resultItem.rightBottom.second = item.rect.y * scaleY + item.rect.height * scaleY;
				resultItem.center_x = item.rect.x * scaleX + item.rect.width * scaleX / 2;
				resultItem.center_y = item.rect.y * scaleY + item.rect.height * scaleY / 2;
				resultItem.area = item.rect.width * scaleX * item.rect.height * scaleY;
				resultItem.classId = item.class_id;
				resultItem.score = item.conf;
				result.push_back(resultItem);
			}
			return result;
		}

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_det::convertWhenLetterBox(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());

			float scale = letterBoxScale;
			int dw = letterBoxdw;
			int dh = letterBoxdh;

			for (const auto& item : detections)
			{
				DetectionRectangleInfo resultItem;

				float x1 = (item.rect.x - dw) / scale;
				float y1 = (item.rect.y - dh) / scale;
				float x2 = (item.rect.x + item.rect.width - dw) / scale;
				float y2 = (item.rect.y + item.rect.height - dh) / scale;

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

		std::vector<DetectionRectangleInfo> ModelEngine_Yolov11_det::convertWhenCentralCrop(
			const std::vector<Detection>& detections)
		{
			std::vector<DetectionRectangleInfo> result;
			result.reserve(detections.size());

			const auto& params = _centerCropParams;

			for (const auto& item : detections)
			{
				float x1 = item.rect.x + params.crop_x - params.pad_left;
				float y1 = item.rect.y + params.crop_y - params.pad_top;
				float x2 = item.rect.x + item.rect.width + params.crop_x - params.pad_left;
				float y2 = item.rect.y + item.rect.height + params.crop_y - params.pad_top;

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
	}
}