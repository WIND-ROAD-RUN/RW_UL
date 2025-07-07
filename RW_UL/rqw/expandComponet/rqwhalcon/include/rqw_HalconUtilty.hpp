#pragma once

#include<QImage>

#include "halconcpp/HalconCpp.h"
#include"opencv2/opencv.hpp"

namespace rw {
	namespace rqw {
		class HalconImageConverter
		{
		public:
			static HalconCpp::HImage QImageToHImage(const QImage& qImage);
			static HalconCpp::HImage CVMatToHImage(const cv::Mat& mat);

		};
	} // namespace rqw
} // namespace rw