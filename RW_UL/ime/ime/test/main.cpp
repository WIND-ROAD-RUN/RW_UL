#include"imeo_ModelEngineFactory_OnnxRuntime.hpp"
#include"ime_ModelEngineFactory.h"

int main()
{
	rw::ModelEngineConfig config;
	config.imagePretreatmentPolicy = rw::ImagePretreatmentPolicy::AdaptiveResize;
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
//#include <opencv2/opencv.hpp>
//
//// letterbox函数：等比例缩放+填充，参数改为引用
//cv::Mat letterbox(const cv::Mat& src, int target_w, int target_h, cv::Scalar color,
//    float& out_scale, int& out_dw, int& out_dh)
//{
//    int src_w = src.cols;
//    int src_h = src.rows;
//    float r = std::min(target_w / (float)src_w, target_h / (float)src_h);
//    int new_unpad_w = int(round(src_w * r));
//    int new_unpad_h = int(round(src_h * r));
//    int dw = (target_w - new_unpad_w) / 2;
//    int dh = (target_h - new_unpad_h) / 2;
//
//    cv::Mat resized;
//    cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h));
//
//    cv::Mat out(target_h, target_w, src.type(), color);
//    resized.copyTo(out(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));
//
//    out_scale = r;
//    out_dw = dw;
//    out_dh = dh;
//    return out;
//}
//
//// 用法示例
//int main() {
//    cv::Mat mat = cv::imread(R"(C:\Users\rw\Desktop\1.png)");
//    int input_w = 640, input_h = 640;
//    float scale;
//    int dw, dh;
//    cv::Mat letterbox_img = letterbox(mat, input_w, input_h, cv::Scalar(0, 0, 0), scale, dw, dh);
//
//    // 归一化和通道变换
//    auto infer_image = cv::dnn::blobFromImage(letterbox_img, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
//
//    // 显示letterbox后的图片
//    cv::imshow("letterbox", letterbox_img);
//    cv::waitKey(0);
//    return 0;
//}