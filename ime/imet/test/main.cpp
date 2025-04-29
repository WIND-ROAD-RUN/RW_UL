#include"opencv2/opencv.hpp"

#include"NvInfer.h"

#include"imet_ModelEngine.h"

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		//// Only output logs with severity greater than warning
		//if (severity <= Severity::kERROR)
		//	std::cout << msg << std::endl;
	}
}logger;

int main() {
	rw::imet::ModelEngine_yolov11_obb model_engine(R"(C:\Users\zfkj\Desktop\yolo11_trt\build\yolo11s.engine)", logger);

	return 0;
}