#include "YOLOv11.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>
#include<opencv2/opencv.hpp>
#define _UNICODE
#include<tchar.h>
#define warmup true


YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger, bool trt)
{
	this->trt = trt;
	init(model_path, logger);
}


void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
	if (trt)
	{
		std::cout << "trt infer" << std::endl;
		//tensorrt
		// Read the engine file
		ifstream engineStream(engine_path, ios::binary);
		engineStream.seekg(0, ios::end);
		const size_t modelSize = engineStream.tellg();
		engineStream.seekg(0, ios::beg);
		unique_ptr<char[]> engineData(new char[modelSize]);
		engineStream.read(engineData.get(), modelSize);
		engineStream.close();

		// Deserialize the tensorrt engine
		runtime = createInferRuntime(logger);
		engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
		context = engine->createExecutionContext();
		
		input_h = engine->getTensorShape(engine->getIOTensorName(0)).d[2];
		input_w = engine->getTensorShape(engine->getIOTensorName(0)).d[3];
		detection_attribute_size = engine->getTensorShape(engine->getIOTensorName(1)).d[1];
		num_detections = engine->getTensorShape(engine->getIOTensorName(1)).d[2];
		num_classes = detection_attribute_size - 4;

		// Initialize input buffers
		cpu_output_buffer = new float[detection_attribute_size * num_detections];
		(cudaMalloc((void**)&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
		// Initialize output buffer
		(cudaMalloc((void**)&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

		if (warmup) {
			for (int i = 0; i < 10; i++) {
				this->infer();
			}
			printf("model warmup 10 times\n");
			cudaDeviceSynchronize();
		}
	}
	else
	{
		std::cout << "onnxruntime infer" << std::endl;
		//onnxruntime
		env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "yolo");
		Ort::SessionOptions options;
		OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0);
		//OrtCUDAProviderOptions cudaOptions;
		//options.AppendExecutionProvider_CUDA(cudaOptions);
		const wchar_t* path = L"yolo11n.onnx";
		session = Ort::Session(env, path, options);
		Ort::AllocatorWithDefaultOptions allocator;
		auto input_name=session.GetInputNameAllocated(0, allocator);
		auto output_name =session.GetOutputNameAllocated(0, allocator);
		
		//auto input_name = session.GetInputName(0, allocator);
		//auto output_name = session.GetOutputName(0, allocator);
		//input_node_names.push_back(input_name.get());
		//output_node_names.push_back(output_name.get());
		input_node_names.push_back("images");
		output_node_names.push_back("output0");
		auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		auto output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		input_h = input_shape[2];
		input_w = input_shape[3];
		detection_attribute_size = output_shape[1];
		num_detections = output_shape[2];
		num_classes = output_shape[1] - 4;
		//if (warmup) {
		//	cv::Mat zero_mat = cv::Mat::zeros(input_h, input_w, CV_8UC3);
		//	preprocess(zero_mat);
		//	for (int i = 0; i < 10; i++) {
		//		this->infer();
		//	}
		//	printf("model warmup 10 times\n");
		//}
	}

}

YOLOv11::~YOLOv11()
{
	if (trt)
	{
		for (int i = 0; i < 2; i++)
			(cudaFree(gpu_buffers[i]));
		delete[] cpu_output_buffer;

		// Destroy the engine
		//cuda_preprocess_destroy();
		delete context;
		delete engine;
		delete runtime;
	}
	else
	{
		output_tensors[0].release();
	}
}

void YOLOv11::preprocess(Mat& image) {

	//// Preprocessing data on gpu
	infer_image = cv::dnn::blobFromImage(image, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true);

	if (trt)
	{
		//tensorrt
		(cudaMemcpy(gpu_buffers[0], infer_image.data, input_w * input_h * image.channels() * sizeof(float), cudaMemcpyHostToDevice));
	}
	else
	{
		// onnxruntime
		// 获取输入节点信息
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
}

void YOLOv11::infer()
{
	if (trt)
	{
		this->context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
		this->context->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
		this->context->enqueueV3(NULL);
	}
	else
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
}

void YOLOv11::postprocess(vector<Detection>& output)
{
	if (trt)
	{
		(cudaMemcpy(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost));
	}
	else
	{
		cpu_output_buffer = output_tensors[0].GetTensorMutableData<float>();
	}
	// Memcpy from device output buffer to host output buffer
	vector<Rect> boxes;
	vector<int> class_ids;
	vector<float> confidences;

	const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

	for (int i = 0; i < det_output.cols; ++i) {
		const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
		Point class_id_point;
		double score;
		minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

		if (score > conf_threshold) {
			const float cx = det_output.at<float>(0, i);
			const float cy = det_output.at<float>(1, i);
			const float ow = det_output.at<float>(2, i);
			const float oh = det_output.at<float>(3, i);
			Rect box;
			box.x = static_cast<int>((cx - 0.5 * ow));
			box.y = static_cast<int>((cy - 0.5 * oh));
			box.width = static_cast<int>(ow);
			box.height = static_cast<int>(oh);

			boxes.push_back(box);
			class_ids.push_back(class_id_point.y);
			confidences.push_back(score);
		}
	}

	vector<int> nms_result;
	dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

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

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
	const float ratio_h = input_h / (float)image.rows;
	const float ratio_w = input_w / (float)image.cols;
	for (int i = 0; i < output.size(); i++)
	{
		auto detection = output[i];
		auto box = detection.bbox;
		auto class_id = detection.class_id;
		auto conf = detection.conf;
		cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);
		box.x = box.x / ratio_w;
		box.y = box.y / ratio_h;
		box.width = box.width / ratio_w;
		box.height = box.height / ratio_h;
		rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);
		// Detection box text
		string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
		Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
		Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
		rectangle(image, text_rect, color, FILLED);
		putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
	}
}