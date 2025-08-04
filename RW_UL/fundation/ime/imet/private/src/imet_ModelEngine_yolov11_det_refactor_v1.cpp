#include"imet_ModelEngine_yolov11_det_refactor_v1.hpp"

#include"cuda_device_runtime_api.h"

#include<fstream>
#include<memory>
#include <iomanip>
#include <sstream>

#include "imet_PreProcess.cuh"

namespace rw
{
	namespace imet
	{
		ModelEngine_yolov11_det_refactor_v1::ModelEngine_yolov11_det_refactor_v1(const std::string& modelPath,
			nvinfer1::ILogger& logger)
		{
			init(modelPath, logger);
		}

		ModelEngine_yolov11_det_refactor_v1::~ModelEngine_yolov11_det_refactor_v1()
		{
			destroy_engineRuntime();
			destroy_buffer();
			destroy_cfg();
		}

		void ModelEngine_yolov11_det_refactor_v1::init(const std::string& enginePath, nvinfer1::ILogger& logger)
		{
			init_engineRuntime(enginePath, logger);
			init_shapeInfo();
			init_buffer();
			ini_cfg();
			warm_up();
		}

		void ModelEngine_yolov11_det_refactor_v1::init_engineRuntime(const std::string& enginePath, nvinfer1::ILogger& logger)
		{
			// Load the engine from the file
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
			cudaStreamCreate(&_stream);
		}

		void ModelEngine_yolov11_det_refactor_v1::destroy_engineRuntime()
		{
			delete _context;
			delete _engine;
			delete _runtime;
			cudaStreamDestroy(_stream);
		}

		void ModelEngine_yolov11_det_refactor_v1::init_shapeInfo()
		{
			_inputShape = _engine->getTensorShape(_engine->getIOTensorName(InputShapeIndexForYolov11));
			_outputShape = _engine->getTensorShape(_engine->getIOTensorName(OutputShapeIndexForYolov11));
			_classNum = _outputShape.d[1] - 4;
			_detectionsNum = _outputShape.d[2];
			_inputHeight = _inputShape.d[2];
			_inputWidth = _inputShape.d[3];
			_channelsNum = _inputShape.d[1];
			size_t input = 1;
			for (int i = 0; i < _inputShape.nbDims; i++)
			{
				input *= _inputShape.d[i];
			}
			_inputSize = input;

			size_t outPut = 1;
			for (int i = 0; i < _outputShape.nbDims; i++)
			{
				outPut *= _outputShape.d[i];
			}
			_outputSize = outPut;

		}

		void ModelEngine_yolov11_det_refactor_v1::init_buffer()
		{
			_hostOutputBuffer= new float[_outputSize];
			cudaMalloc(reinterpret_cast<void**>(&_deviceInputBuffer), _inputSize * sizeof(float));
			cudaMalloc(reinterpret_cast<void**>(&_deviceOutputBuffer), _outputSize * sizeof(float));
			cudaMalloc(reinterpret_cast<void**>(&_deviceTransposeBuffer), _outputSize * sizeof(float));
			cudaMalloc(reinterpret_cast<void**>(&_deviceDecodeBuffer), (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float));
			_context->setInputTensorAddress(_engine->getIOTensorName(InputShapeIndexForYolov11), _deviceInputBuffer);
			_context->setOutputTensorAddress(_engine->getIOTensorName(OutputShapeIndexForYolov11), _deviceOutputBuffer);
			_hostOutputBuffer1 = new float[1 + kMaxNumOutputBbox * kNumBoxElement];

		}

		void ModelEngine_yolov11_det_refactor_v1::destroy_buffer()
		{
			delete[] _hostOutputBuffer;
			delete[] _hostOutputBuffer1;
			cudaFree(_deviceInputBuffer);
			cudaFree(_deviceOutputBuffer);
			cudaFree(_deviceTransposeBuffer);
			cudaFree(_deviceDecodeBuffer);
		}

		void ModelEngine_yolov11_det_refactor_v1::ini_cfg()
		{
			cudaMalloc((void**)&_deviceClassIdNmsTogether, _config.classids_nms_together.size() * sizeof(size_t));
		}

		void ModelEngine_yolov11_det_refactor_v1::destroy_cfg()
		{
			cudaFree(_deviceClassIdNmsTogether);
		}

		void ModelEngine_yolov11_det_refactor_v1::warm_up()
		{
			for (int i=0;i<20;i++)
			{
				this->infer();
			}
			cudaDeviceSynchronize();
		}

		void ModelEngine_yolov11_det_refactor_v1::preprocess(const cv::Mat& mat)
		{
			_sourceImgWidth = mat.cols;
			_sourceImgHeight = mat.rows;

			unsigned char pad_b = static_cast<unsigned char>(_config.letterBoxColor[0]);
			unsigned char pad_g = static_cast<unsigned char>(_config.letterBoxColor[1]);
			unsigned char pad_r = static_cast<unsigned char>(_config.letterBoxColor[2]);

			LetterBoxConfig cfg;
			cfg.dstDevData = _deviceInputBuffer;
			cfg.dstHeight = _inputHeight;
			cfg.dstWidth = _inputWidth;
			cfg.pad_b = pad_b;
			cfg.pad_g = pad_g;
			cfg.pad_r = pad_r;
			_letterBoxInfo = ImgPreprocess::LetterBox(mat, cfg, _stream);

			//testCode
			{
				//// 1. 分配主机内存
			//std::vector<float> cpuInput(_inputSize);
			//cudaMemcpy(cpuInput.data(), _deviceInputBuffer, _inputSize * sizeof(float), cudaMemcpyDeviceToHost);

			//// 2. 转换为Mat（假设为BGR、float、NCHW）
			//int channels = _channelsNum;
			//int height = _inputHeight;
			//int width = _inputWidth;

			//// 3. NCHW -> HWC
			//cv::Mat img(height, width, CV_32FC3);
			//for (int c = 0; c < channels; ++c) {
			//	for (int h = 0; h < height; ++h) {
			//		for (int w = 0; w < width; ++w) {
			//			img.at<cv::Vec3f>(h, w)[c] = cpuInput[c * height * width + h * width + w];
			//		}
			//	}
			//}

			//// 4. 若有归一化，反归一化
			//img *= 255.0f;
			//img.convertTo(img, CV_8UC3);

			//// 5. 显示或保存
			//cv::imshow("cuda_preprocessed", img);
			//cv::waitKey(0);
			//// 或 cv::imwrite("cuda_preprocessed.jpg", img);
			}
		}

		void ModelEngine_yolov11_det_refactor_v1::infer()
		{
			this->_context->enqueueV3(_stream);
			Utility::transpose(_deviceOutputBuffer, _deviceTransposeBuffer, _outputShape.d[1], _outputShape.d[2], _stream);
			Utility::decode(
				_deviceTransposeBuffer,
				_deviceDecodeBuffer, 
				_detectionsNum, 
				_classNum,
				_config.conf_threshold, 
				kMaxNumOutputBbox,
				kNumBoxElement,
				_stream);

			{
				float* hostDecodeBuffer = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
				cudaMemcpy(hostDecodeBuffer, _deviceDecodeBuffer, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost);
				int num = std::min((int)hostDecodeBuffer[0], kMaxNumOutputBbox);
				std::vector<std::vector<float>> decodeVec;
				decodeVec.reserve(num);

				for (int i = 0; i < num; ++i) {
					float* item = hostDecodeBuffer + 1 + i * kNumBoxElement;
					std::vector<float> row(item, item + kNumBoxElement);
					decodeVec.push_back(std::move(row));
				}

			}

			Utility::nms(_deviceDecodeBuffer, _config.nms_threshold, kMaxNumOutputBbox, kNumBoxElement, _deviceClassIdNmsTogether, _config.classids_nms_together.size(), _stream);
			cudaMemcpyAsync(_hostOutputBuffer1, _deviceDecodeBuffer, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost, _stream);
			cudaStreamSynchronize(_stream);


			{
				std::vector<DetectionRectangleInfo> ret;
				std::vector<Detection> vDetections;
				int count = std::min((int)_hostOutputBuffer1[0], kMaxNumOutputBbox);
				/*for (int i = 0; i < count; i++) {
					int pos = 1 + i * kNumBoxElement;
					int keepFlag = (int)outputData[pos + 6];
					if (keepFlag == 1) {
						Detection det;
						memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
						det.conf = outputData[pos + 4];
						det.classId = (int)outputData[pos + 5];

						vDetections.emplace_back(det);
					}
				}*/
			}

		}

		std::vector<DetectionRectangleInfo> ModelEngine_yolov11_det_refactor_v1::postProcess()
		{
			std::vector<Detection> output;
			(cudaMemcpy(_hostOutputBuffer, _deviceOutputBuffer, _outputSize * sizeof(float), cudaMemcpyDeviceToHost));
			std::vector<cv::Rect> boxes;
			std::vector<int> class_ids;
			std::vector<float> confidences;


			const cv::Mat det_output(_classNum + 4, _detectionsNum, CV_32F, _hostOutputBuffer);
			

			for (int i = 0; i < det_output.cols; ++i) {
				const  cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + _classNum);
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

			std::cout << "size:" << size << std::endl;
			//auto result = convertDetectionToDetectionRectangleInfo(output);

			return std::vector<DetectionRectangleInfo>();
		}
	}
}
