#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <iostream>
#include <string>
#include<opencv2/videoio.hpp>
#include<opencv2/opencv.hpp>
using namespace cv;

#include"imeo_ModelEngine_yolov11_obb.hpp"

int main()
{
	rw::imeo::ModelEngine_yolov11_obb model("D:/Workplace/rep/RW_UL/Project/yolo/build/yolo11n.onnx");
	cv::Mat image = cv::imread("D:/Workplace/rep/RW_UL/Project/yolo/build/bus.jpg");
	if (image.empty())
	{
		std::cerr << "Error reading image: " << "D:/yolo/build/bus.jpg" << std::endl;
	}
	auto result= model.processImg(image);
	image=model.draw(image, result);
	cv::imshow("sad", image);
	cv::waitKey(0);
}
