#include"ime_utilty.hpp"

namespace rw
{
	cv::Mat ImagePainter::drawShapes(const cv::Mat& image, const std::vector<DetectionRectangleInfo>& rectInfo,
		PainterConfig config)
	{
		cv::Mat resultImage = image.clone();
		for (const auto& rect : rectInfo) {
			drawShapesOnSourceImg(resultImage, rect, config);
		}
		return resultImage;
	}

	void ImagePainter::drawShapesOnSourceImg(cv::Mat& image, const std::vector<DetectionRectangleInfo>& rectInfo,
		PainterConfig config)
	{
        for (const auto& rect : rectInfo) {
            drawShapesOnSourceImg(image, rect, config);
        }
	}

	cv::Mat ImagePainter::drawShapes(
        const cv::Mat& image,
        const DetectionRectangleInfo& rectInfo,
        PainterConfig config
    ) {
        cv::Mat resultImage = image.clone();

        drawShapesOnSourceImg(resultImage, rectInfo, config);

        return resultImage;
    }

    void ImagePainter::drawShapesOnSourceImg(cv::Mat& image, const DetectionRectangleInfo& rectInfo,
	    PainterConfig config)
    {
        if (config.shapeType == ShapeType::Rectangle) {
            cv::rectangle(
                image,
                cv::Point(static_cast<int>(rectInfo.leftTop.first), static_cast<int>(rectInfo.leftTop.second)),
                cv::Point(static_cast<int>(rectInfo.rightBottom.first), static_cast<int>(rectInfo.rightBottom.second)),
                config.color,
                config.thickness
            );
        }
        else if (config.shapeType == ShapeType::Circle) {
 
            cv::Point center(rectInfo.center_x, rectInfo.center_y);

            int radius = std::min(rectInfo.width, rectInfo.height) / 2;

            cv::circle(
                image,
                center,
                radius,
                config.color, 
                config.thickness
            );
        }


        cv::putText(
            image,
            config.text, 
            cv::Point(static_cast<int>(rectInfo.leftTop.first), static_cast<int>(rectInfo.leftTop.second) - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            config.fontSize / 10.0, 
            config.textColor,
            config.fontThickness
        );
    }

    void ImagePainter::drawVerticalLine(cv::Mat& image, int position, const ImagePainter::PainterConfig& config)
    {
        if (image.empty()) {
            return;
        }

        if (position < 0 || position >= image.cols) {
            return;
        }

        cv::Scalar lineColor = config.color;
        int thickness = config.thickness;

        cv::line(image, cv::Point(position, 0), cv::Point(position, image.rows - 1), lineColor, thickness);
    }

    void ImagePainter::drawHorizontalLine(cv::Mat& image, int position, const ImagePainter::PainterConfig& config)
    {
        // 检查图像是否为空
        if (image.empty()) {
            return;
        }

        // 检查位置是否在图像范围内
        if (position < 0 || position >= image.rows) {
            return;
        }

        // 设置线条颜色和粗细
        cv::Scalar lineColor = config.color;
        int thickness = config.thickness;

        // 绘制水平线
        cv::line(image, cv::Point(0, position), cv::Point(image.cols - 1, position), lineColor, thickness);
    }
}
