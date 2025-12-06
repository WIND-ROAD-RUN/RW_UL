#include <numeric>

#include"opencv2/opencv.hpp"

#include"NvInfer.h"
#include"imet_ModelEngine_yolov11_det.hpp"
#include<string>

#include <filesystem>
#include <Windows.h>

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
	// ========== 设置控制台编码为 UTF-8 ==========
	SetConsoleOutputCP(CP_UTF8);
	setvbuf(stdout, nullptr, _IOFBF, 1000);
	// ========== 配置模型和图片路径 ==========
	std::string modelPath = R"(D:\zfkjDevelopment\ThirdLibrary\TensorRTs\cuda11\8.6\bin\HandleScanner.engine)";
	std::string testImageFolder = R"(C:\Users\zfkj4090\Downloads\1\images\train)";

	// ========== 创建模型引擎 (TensorRT 8.6) ==========
	rw::imet::ModelEngine_Yolov11_det modelEngine(modelPath, logger);

	// ========== 配置模型参数 ==========
	rw::ModelEngineConfig config;
	config.conf_threshold = 0.25f;
	config.nms_threshold = 0.45f;
	config.imagePretreatmentPolicy = rw::ImagePretreatmentPolicy::LetterBox;
	config.letterBoxColor = cv::Scalar(114, 114, 114);
	modelEngine.setConfig(config);

	// ========== 收集所有图片文件 ==========
	std::vector<fs::path> imageFiles;
	std::vector<std::string> supportedExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif" };

	try {
		for (const auto& entry : fs::directory_iterator(testImageFolder)) {
			if (entry.is_regular_file()) {
				std::string ext = entry.path().extension().string();
				std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

				if (std::find(supportedExtensions.begin(), supportedExtensions.end(), ext) != supportedExtensions.end()) {
					imageFiles.push_back(entry.path());
				}
			}
		}
	}
	catch (const fs::filesystem_error& e) {
		std::cerr << "读取文件夹失败: " << e.what() << std::endl;
		return -1;
	}

	if (imageFiles.empty()) {
		std::cerr << "文件夹中没有找到支持的图片文件" << std::endl;
		return -1;
	}

	std::cout << "========== 找到 " << imageFiles.size() << " 张图片 ==========" << std::endl;
	std::cout << std::endl;

	// ========== 处理所有图片 ==========
	std::vector<double> processingTimes;
	int successCount = 0;
	int failCount = 0;

	for (size_t idx = 0; idx < imageFiles.size(); ++idx) {
		const auto& imagePath = imageFiles[idx];
		std::string imagePathStr = imagePath.string();
		std::string imageName = imagePath.filename().string();

		std::cout << "========== [" << (idx + 1) << "/" << imageFiles.size() << "] 处理: " << imageName << " ==========" << std::endl;

		// 读取图片
		cv::Mat testMat = cv::imread(imagePathStr);
		if (testMat.empty()) {
			std::cerr << "无法读取图片: " << imagePathStr << std::endl;
			failCount++;
			std::cout << std::endl;
			continue;
		}

		// 记录开始时间
		auto startTime = std::chrono::high_resolution_clock::now();

		// 处理图片
		auto result = modelEngine.processImg(testMat);

		// 记录结束时间
		auto endTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> duration = endTime - startTime;
		double processingTimeMs = duration.count();
		processingTimes.push_back(processingTimeMs);

		// 打印结果
		std::cout << "处理时间: " << std::fixed << std::setprecision(2) << processingTimeMs << " ms" << std::endl;
		std::cout << "检测到 " << result.size() << " 个目标" << std::endl;

		for (size_t i = 0; i < result.size(); ++i) {
			const auto& det = result[i];
			std::cout << "  [" << i << "] classId=" << det.classId
				<< " score=" << std::fixed << std::setprecision(3) << det.score
				<< " center=(" << det.center_x << "," << det.center_y << ")"
				<< " size=" << det.width << "x" << det.height << std::endl;
		}

		successCount++;
		std::cout << std::endl;

		// 可选：显示结果（按ESC跳过显示，按其他键继续查看）
		/*
		auto drawImg = modelEngine.draw(testMat, result);
		cv::Mat displayImg;
		int maxWidth = 1280;
		int maxHeight = 720;

		double scaleW = static_cast<double>(maxWidth) / drawImg.cols;
		double scaleH = static_cast<double>(maxHeight) / drawImg.rows;
		double scale = std::min({ scaleW, scaleH, 1.0 });

		if (scale < 1.0) {
			cv::resize(drawImg, displayImg, cv::Size(), scale, scale, cv::INTER_AREA);
			cv::imshow("Detection Result", displayImg);
		}
		else {
			cv::imshow("Detection Result", drawImg);
		}

		int key = cv::waitKey(0);
		if (key == 27) { // ESC键退出显示
			cv::destroyAllWindows();
		}
		*/
	}

	// ========== 打印统计信息 ==========
	std::cout << "========================================" << std::endl;
	std::cout << "========== 处理完成统计 ==========" << std::endl;
	std::cout << "总图片数: " << imageFiles.size() << std::endl;
	std::cout << "成功处理: " << successCount << std::endl;
	std::cout << "处理失败: " << failCount << std::endl;

	if (!processingTimes.empty()) {
		double totalTime = std::accumulate(processingTimes.begin(), processingTimes.end(), 0.0);
		double avgTime = totalTime / processingTimes.size();
		double minTime = *std::min_element(processingTimes.begin(), processingTimes.end());
		double maxTime = *std::max_element(processingTimes.begin(), processingTimes.end());

		std::cout << std::endl;
		std::cout << "时间统计:" << std::endl;
		std::cout << "  总时间: " << std::fixed << std::setprecision(2) << totalTime << " ms" << std::endl;
		std::cout << "  平均时间: " << std::fixed << std::setprecision(2) << avgTime << " ms" << std::endl;
		std::cout << "  最小时间: " << std::fixed << std::setprecision(2) << minTime << " ms" << std::endl;
		std::cout << "  最大时间: " << std::fixed << std::setprecision(2) << maxTime << " ms" << std::endl;
		std::cout << "  平均FPS: " << std::fixed << std::setprecision(2) << (1000.0 / avgTime) << std::endl;
	}
	std::cout << "========================================" << std::endl;

	return 0;
}