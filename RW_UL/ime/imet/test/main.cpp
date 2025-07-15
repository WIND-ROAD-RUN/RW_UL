#include"opencv2/opencv.hpp"

#include"NvInfer.h"
#include"imet_ModelEngine_yolov11_seg_refacotr.hpp"
#include"imet_ModelEngine_yolov11_seg.hpp"
#include"imet_ModelEngineFactory_TensorRT.hpp"
#include<string>

using namespace std;
using namespace cv;


int main() {
    rw::ModelEngineConfig config;
    config.modelPath = R"(C:\Users\rw\Desktop\models\niukou.engine)";
    auto modelEngine = rw::imet::ModelEngineFactory_TensorRT::createModelEngine(config, rw::ModelType::Yolov11_Seg_with_mask);
    auto mat = cv::imread(R"(C:\Users\rw\Desktop\c3a0dd3937b5a61270469cb21491d6a7.jpg)");

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();
    auto result = modelEngine->processImg(mat);
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算运行时间
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "processImg 运行时间: " << elapsed.count() << " 毫秒" << std::endl;

    auto matResult = modelEngine->draw(mat, result);
    cv::imshow("result", matResult);
    cv::waitKey(0);

    for (const auto & item:result)
    {
		std::cout << "area: " << item.area << std::endl;
    }

    return 0;
}