#include"ime_ModelEngine.h"

namespace rw
{
	void ModelEngine::setDrawStatus(bool status)
	{
		isDraw = status;
	}

	cv::Mat ModelEngine::draw(const cv::Mat& mat)
	{
		return mat;
	}

	DetectionRectangleInfo ModelEngine::processImg(const cv::Mat& mat)
	{
		preprocess(mat);
		infer();
		DetectionRectangleInfo detection = postProcess();
		return detection;
	}

	cv::Mat ModelEngine::processImg(const cv::Mat& mat, DetectionRectangleInfo& detection)
	{
		preprocess(mat);
		infer();
		detection = postProcess();
		if (isDraw)
		{
			return draw(mat);
		}
		return mat;
	}
}
