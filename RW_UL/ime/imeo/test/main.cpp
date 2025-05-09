#include"imeo_ModelEngineFactory_OnnxRuntime.hpp"

int main()
{
	rw::ModelEngineConfig config;
	config.modelPath=R"(D:\Workplace\rep\RW_UL\Project\yolo\build\yolo11n.onnx)";
	auto model=rw::imeo::ModelEngineFactory_OnnxRuntime::createModelEngine(config,rw::ModelType::yolov11_obb);
	cv::Mat image = cv::imread("D:/Workplace/rep/RW_UL/Project/yolo/build/bus.jpg");
	if (image.empty())
	{
		std::cerr << "Error reading image: " << "D:/yolo/build/bus.jpg" << std::endl;
	}
	auto result= model->processImg(image);
	image=model->draw(image, result);
	cv::imshow("sad", image);
	cv::waitKey(0);
}
