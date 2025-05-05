#include"opencv2/opencv.hpp"

#include"ime_ModelEngineFactory.h"
#include<string>

using namespace std;
using namespace cv;

int main()
{
	const string path{ R"(D:\zfkjData\ButtonScanner\ModelStorage\Temp\Image\work1\bad\20250426155302818.png)" };

	rw::ModelEngineConfig config;
	config.ModelPath = R"(C:\Users\rw\Desktop\model\best.engine)";

	auto modelEngine = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::yolov11_obb,rw::ModelEngineDeployType::TensorRT);
	Mat image = imread(path);
	if (image.empty())
	{
		cerr << "error reading image: " << path << endl;
	}

	std::vector<rw::DetectionRectangleInfo> detection;
	modelEngine->setDrawStatus(true);

	auto start = std::chrono::system_clock::now();
	auto result = modelEngine->processImg(image, detection);
	auto end = std::chrono::system_clock::now();

	auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
	printf("cost %2.4lf ms\n", tc);

	cv::imshow("asd", result);
	cv::waitKey(0);
}