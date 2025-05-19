#include"opencv2/opencv.hpp"

#include"NvInfer.h"
#include"ime_ModelEngineFactory.h"
#include<string>

using namespace std;
using namespace cv;

int main() {
	rw::ModelEngineConfig config;
	config.conf_threshold = 0.2f;
	config.nms_threshold = 0.1f;
	config.modelPath = R"(C:\Users\rw\Desktop\yolo11n-obb.engine)";
	auto model_engine = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::yolov11_obb,rw::ModelEngineDeployType::TensorRT);

	const string path{ R"(C:\Users\rw\Desktop\car.jpg)" };

	Mat image = imread(path);
	if (image.empty())
	{
		cerr << "error reading image: " << path << endl;
	}
	auto start = std::chrono::system_clock::now();
	auto result = model_engine->processImg(image);
	auto resultImage = model_engine->draw(image, result);
	auto end = std::chrono::system_clock::now();

	auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
	printf("cost %2.4lf ms\n", tc);

	imshow("result", resultImage);

	cv::waitKey(0);
	return 0;
}