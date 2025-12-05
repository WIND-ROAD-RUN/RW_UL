#include <numeric>

#include"opencv2/opencv.hpp"

#include"NvInfer.h"
#include"imet_ModelEngine_yolov11_det.hpp"
#include<string>

#include <filesystem>

#include "imet_ModelEngine_yolov11_det_cudaAcc.hpp"
#include "imet_ModelEngine_yolov11_obb.hpp"
#include "imet_ModelEngine_yolov11_seg.hpp"
#include "imet_ModelEngine_yolov11_seg_cudaAcc.hpp"
#include "imet_ModelEngine_yolov11_seg_mask.hpp"
#include "imet_ModelEngine_yolov11_seg_mask_cudaAcc.hpp"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
	}
}logger;

int main() {
	// ========== 配置模型和图片路径 ==========
	std::string modelPath = R"(D:\zfkjDevelopment\ThirdLibrary\TensorRTs\cuda11\8.6\bin\KeyScanner-20251108.engine)";
	std::string testImagePath = R"(C:\Users\zfkj4090\Downloads\3\images\train\OK20251114140111911.jpg)";

	// ========== 创建模型引擎 (TensorRT 8.6) ==========
	rw::imet::ModelEngine_Yolov11_seg modelEngine(modelPath, logger);

	// ========== 配置模型参数 ==========
	rw::ModelEngineConfig config;
	config.conf_threshold = 0.25f;
	config.nms_threshold = 0.45f;
	config.imagePretreatmentPolicy = rw::ImagePretreatmentPolicy::LetterBox;
	config.letterBoxColor = cv::Scalar(114, 114, 114);
	modelEngine.setConfig(config);

	// ========== 单张图片测试 ==========
	std::cout << "========== 单张图片测试 ==========" << std::endl;
	cv::Mat testMat = cv::imread(testImagePath);
	if (testMat.empty()) {
		std::cerr << "无法读取测试图片: " << testImagePath << std::endl;
		return -1;
	}

	auto result = modelEngine.processImg(testMat);
	auto drawImg = modelEngine.draw(testMat, result);

	std::cout << "检测到 " << result.size() << " 个目标" << std::endl;
	for (size_t i = 0; i < result.size(); ++i) {
		const auto& det = result[i];
		std::cout << "  [" << i << "] classId=" << det.classId
			<< " score=" << std::fixed << std::setprecision(3) << det.score
			<< " center=(" << det.center_x << "," << det.center_y << ")"
			<< " size=" << det.width << "x" << det.height << std::endl;
	}

	cv::Mat displayImg;
	int maxWidth = 1280;  // 最大宽度
	int maxHeight = 720;  // 最大高度

	double scaleW = static_cast<double>(maxWidth) / drawImg.cols;
	double scaleH = static_cast<double>(maxHeight) / drawImg.rows;
	double scale = std::min({ scaleW, scaleH, 1.0 }); // 取最小值,且不放大

	if (scale < 1.0) {
		cv::resize(drawImg, displayImg, cv::Size(), scale, scale, cv::INTER_AREA);
		cv::imshow("Detection Result", displayImg);
	}
	else {
		cv::imshow("Detection Result", drawImg);
	}
	cv::waitKey(0);


    return 0;
}