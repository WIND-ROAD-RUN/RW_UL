#include"imeo_ModelEngineFactory_OnnxRuntime.hpp"
#include"ime_ModelEngineFactory.h"

int main()
{
	rw::ModelEngineConfig config;
	config.modelPath = R"(C:\Users\rw\Desktop\models\SegModel.engine)";
	auto model = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::yolov11_seg,rw::ModelEngineDeployType::TensorRT);
	cv::Mat image = cv::imread(R"(C:\Users\rw\Desktop\NG\NG20250517091144654.png)");
	if (image.empty())
	{
		std::cerr << "Error reading image: " << "D:/yolo/build/bus.jpg" << std::endl;
	}
	auto result = model->processImg(image);
	image = model->draw(image, result);
	cv::imshow("sad", image);
	cv::waitKey(0);
}
