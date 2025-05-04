#include"opencv2/opencv.hpp"

#include"NvInfer.h"

#include"imet_ModelEngine_yolov11_obb.hpp"
#include"imet_ModelEngine_yolov11_seg.hpp"
#include<string>
using namespace std;
using namespace cv;
class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		//// Only output logs with severity greater than warning
		//if (severity <= Severity::kERROR)
		//	std::cout << msg << std::endl;
	}
}logger;

//int main() {
//	rw::imet::ModelEngine_yolov11_seg model_engine(R"(C:\Users\zfkj\Desktop\model\best_seg.engine)", logger);
//
//	const string path{ R"(C:\Users\zfkj\Desktop\1\images\train\05113212738.jpg)" };
//
//	Mat image = imread(path);
//	if (image.empty())
//	{
//		cerr << "error reading image: " << path << endl;
//	}
//	vector<DetectionSeg> objects;
//	auto start = std::chrono::system_clock::now();
//	model_engine.preprocess(image);
//	model_engine.infer();
//	model_engine.postprocess(objects);
//	model_engine.draw(image, objects);
//	auto end = std::chrono::system_clock::now();
//
//	auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
//	printf("cost %2.4lf ms\n", tc);
//
//	imshow("result", image);
//
//	cv::waitKey(0);
//	return 0;
//
//}

int main() {
	rw::imet::ModelEngine_yolov11_obb model_engine(R"(C:\Users\rw\Desktop\model\best.engine)", logger);

	const string path{ R"(C:\Users\rw\Desktop\1.png)"};

	Mat image = imread(path);
	if (image.empty())
	{
		cerr << "Error reading image: " << path << endl;
	}
	vector<Detection> objects;
	auto start = std::chrono::system_clock::now();
	model_engine.preprocess(image);
	model_engine.infer();
	model_engine.postprocess(objects);
	model_engine.draw(image, objects);
	auto end = std::chrono::system_clock::now();

	auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
	printf("cost %2.4lf ms\n", tc);

	imshow("Result", image);

	waitKey(0);
	return 0;

}