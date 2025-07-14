#pragma once

#include<QImage>
#include<QPixmap>

#include "rqw_rqwColor.hpp"
#include"opencv2/opencv.hpp"

namespace HalconCpp
{
	class HTuple;
	class HObject;
	class HImage;
}


namespace rw {
	namespace rqw {
		using HalconWidgetDisObjectId = int;
		using HalconShapeId = HalconCpp::HTuple;

		HalconCpp::HImage QImageToHImage(const QImage& qImage);
		HalconCpp::HImage CvMatToHImage(const cv::Mat& mat);
		HalconCpp::HImage QPixmapToHImage(const QPixmap& pixmap);

        struct PainterConfig
        {
        public:
            PainterConfig() = default;

            PainterConfig(const RQWColor& color, int thickness)
                : color(color), thickness(thickness) {
            }

            PainterConfig(const PainterConfig& other) = default;

            PainterConfig(PainterConfig&& other) noexcept = default;

            PainterConfig& operator=(const PainterConfig& other) = default;

            PainterConfig& operator=(PainterConfig&& other) noexcept = default;

            ~PainterConfig() = default;
        public:
            RQWColor color{ RQWColor::Black };
            int thickness{ 3 };
        };
	} // namespace rqw
} // namespace rw