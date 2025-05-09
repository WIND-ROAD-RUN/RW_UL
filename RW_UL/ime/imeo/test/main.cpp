#include"onnxruntime_cxx_api.h"

#include"opencv2/opencv.hpp"

#include<iostream>
#include<vector>
#include<string>

#include"imeo_ModelEngine_yolov11_obb.hpp"

int main()
{
	rw::imeo::ModelEngine_yolov11_obb model("C:/Users/rw/Desktop/yolo11n.onnx");
}