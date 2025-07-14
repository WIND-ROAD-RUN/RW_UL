#pragma once

#include<QImage>

#include "halconcpp/HalconCpp.h"
#include"opencv2/opencv.hpp"

namespace rw {
	namespace rqw {
		HalconCpp::HImage QImageToHImage(const QImage& qImage);
		HalconCpp::HImage CvMatToHImage(const cv::Mat& mat);
		HalconCpp::HImage QPixmapToHImage(const QPixmap& pixmap);
	} // namespace rqw
} // namespace rw