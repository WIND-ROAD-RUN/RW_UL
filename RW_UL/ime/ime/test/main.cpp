#include"imeo_ModelEngineFactory_OnnxRuntime.hpp"
#include"ime_ModelEngineFactory.h"

int main()
{
	rw::ModelEngineConfig config;
	config.imagePretreatmentPolicy = rw::ImagePretreatmentPolicy::LetterBox;
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
//struct CenterCropParams {
//    int pad_left, pad_top;
//    int crop_x, crop_y;
//};
//
//static cv::Mat centerCrop(
//    const cv::Mat& src, int target_w, int target_h, 
//    cv::Scalar pad_color,
//    CenterCropParams* out_params = nullptr
//    )
//{
//    int src_w = src.cols;
//    int src_h = src.rows;
//
//    int pad_left = std::max(0, (target_w - src_w) / 2);
//    int pad_right = std::max(0, target_w - src_w - pad_left);
//    int pad_top = std::max(0, (target_h - src_h) / 2);
//    int pad_bottom = std::max(0, target_h - src_h - pad_top);
//
//    cv::Mat padded;
//    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0) {
//        cv::copyMakeBorder(src, padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, pad_color);
//    }
//    else {
//        padded = src;
//    }
//
//    int crop_x = std::max(0, (padded.cols - target_w) / 2);
//    int crop_y = std::max(0, (padded.rows - target_h) / 2);
//
//    if (out_params) {
//        out_params->pad_left = pad_left;
//        out_params->pad_top = pad_top;
//        out_params->crop_x = crop_x;
//        out_params->crop_y = crop_y;
//    }
//
//    cv::Rect roi(crop_x, crop_y, target_w, target_h);
//    return padded(roi).clone();
//}
//
//int main() {
//    cv::Mat mat = cv::imread(R"(C:\Users\rw\Desktop\NG\NG20250517091144654.png)");
//    int input_w = 200, input_h = 200;
//    float scale;
//    int dw, dh;
//    cv::Mat letterbox_img = centerCrop(mat, input_w, input_h, cv::Scalar(255, 255, 255));
//
//    // 归一化和通道变换
//    auto infer_image = cv::dnn::blobFromImage(letterbox_img, 1.f / 255.f, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
//
//    // 显示letterbox后的图片
//    cv::imshow("letterbox", letterbox_img);
//    cv::waitKey(0);
//    return 0;
//}