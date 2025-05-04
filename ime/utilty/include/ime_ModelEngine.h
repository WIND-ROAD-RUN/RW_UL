#pragma once

#include "ime_utilty.hpp"

#include"opencv2/opencv.hpp"

namespace rw
{
	class ModelEngine
	{
	private:
		std::atomic_bool isDraw{false};
	public:
		ModelEngine() = default;
		virtual ~ModelEngine() = default;
	public:
		ModelEngine(const ModelEngine & modelEngine)=delete;
		ModelEngine(ModelEngine&& modelEngine) = delete;
		ModelEngine& operator=(const ModelEngine& modelEngine) = delete;
		ModelEngine& operator=(ModelEngine&& modelEngine) = delete;
	private:
		virtual void preprocess(const cv::Mat& mat)=0;
		virtual void infer() = 0;
		virtual DetectionRectangleInfo postProcess() = 0;
		virtual cv::Mat draw(const cv::Mat& mat);
	public:
		DetectionRectangleInfo processImg(const cv::Mat & mat);
		cv::Mat processImg(const cv::Mat& mat, DetectionRectangleInfo & detection);
		void setDrawStatus(bool status);
	};

}