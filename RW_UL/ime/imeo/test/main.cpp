#include"imeo_ModelEngineFactory_OnnxRuntime.hpp"

int main()
{
	rw::ModelEngineConfig config;
	config.modelPath=R"(C:\Users\rw\Desktop\model\best_seg.onnx)";
	auto model=rw::imeo::ModelEngineFactory_OnnxRuntime::createModelEngine(config,rw::ModelType::yolov11_seg);
	cv::Mat image = cv::imread(R"(C:\Users\rw\Desktop\1.png)");
	if (image.empty())
	{
		std::cerr << "Error reading image: " << "D:/yolo/build/bus.jpg" << std::endl;
	}
	auto result= model->processImg(image);
	image=model->draw(image, result);
	cv::imshow("sad", image);
	cv::waitKey(0);
}
