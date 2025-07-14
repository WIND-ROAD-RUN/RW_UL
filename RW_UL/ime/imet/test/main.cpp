#include"opencv2/opencv.hpp"

#include"NvInfer.h"
#include"imet_ModelEngine_yolov11_seg_refacotr.hpp"
#include"imet_ModelEngine_yolov11_seg.hpp"
#include<string>

using namespace std;
using namespace cv;

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
	}
}logger;

int main() {
	rw::imet::ModelEngine_Yolov11_seg_refactor modelEngine(R"(C:\Users\rw\Desktop\models\niukou.engine)", logger);
	auto mat = cv::imread(R"(C:\Users\rw\Desktop\c3a0dd3937b5a61270469cb21491d6a7.jpg)");
	auto result = modelEngine.processImg(mat);
	auto matResult = modelEngine.draw(mat, result);
	cv::imshow("result", matResult);
	cv::waitKey(0);
	return 0;
}