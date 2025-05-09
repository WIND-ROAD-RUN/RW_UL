#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <iostream>
#include <string>
#include "src/YOLOv11.h"
#include<opencv2/videoio.hpp>
#include<opencv2/opencv.hpp>
using namespace cv;

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		// Only output logs with severity greater than warning
		if (severity <= Severity::kERROR)
			std::cout << msg << std::endl;
	}
}logger;

int main(int argc, char** argv)
{
	bool trt = 0;
	const string engine_file_path{ "yolo11n.engine" };
	//const string engine_file_path{ "yolov11n_int8.engine" };
	const string path{ "bus.jpg" };
	// init model
	YOLOv11 model(engine_file_path, logger,trt);
	// open image
	Mat image = imread(path);
	if (image.empty())
	{
		cerr << "Error reading image: " << path << endl;
	}
	vector<Detection> objects;
	model.preprocess(image);
	auto start = std::chrono::system_clock::now();
	model.infer();
	auto end = std::chrono::system_clock::now();
	model.postprocess(objects);
	model.draw(image, objects);

	auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
	printf("cost %2.4lf ms\n", tc);

	imshow("Result", image);

	waitKey(0);
	return 0;
}