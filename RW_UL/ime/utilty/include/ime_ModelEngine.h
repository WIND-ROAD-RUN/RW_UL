#pragma once

#include "ime_utilty.hpp"

#include"opencv2/opencv.hpp"

#include<vector>

namespace rw
{
	class ModelEngine
	{
	private:
		std::atomic_bool isDraw{ false };
	public:
		ModelEngine() = default;
		virtual ~ModelEngine() = default;
	public:
		ModelEngine(const ModelEngine& modelEngine) = delete;
		ModelEngine(ModelEngine&& modelEngine) = delete;
		ModelEngine& operator=(const ModelEngine& modelEngine) = delete;
		ModelEngine& operator=(ModelEngine&& modelEngine) = delete;
	private:
		virtual void preprocess(const cv::Mat& mat) = 0;
		virtual void infer() = 0;
		virtual std::vector<DetectionRectangleInfo> postProcess() = 0;
	public:
		virtual cv::Mat draw(const cv::Mat& mat, const std::vector<DetectionRectangleInfo>& infoList);
	public:
		std::vector<DetectionRectangleInfo> processImg(const cv::Mat& mat);
		cv::Mat processImg(const cv::Mat& mat, std::vector<DetectionRectangleInfo>& detection);
		void setDrawStatus(bool status);
	};
}