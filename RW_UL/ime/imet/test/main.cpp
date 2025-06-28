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
	rw::imet::ModelEngine_Yolov11_seg_refactor modelEngine(R"(C:\Users\rw\Desktop\models\SegModel.engine)", logger);
	auto mat = cv::imread(R"(D:\zfkjData\ButtonScanner\ModelStorage\Temp\Image\work1\bad\NG20250417152301729.png)");
	auto result = modelEngine.processImg(mat);
	auto matResult = modelEngine.draw(mat, result);
	cv::imshow("result", matResult);
	cv::waitKey(0);
	return 0;
}