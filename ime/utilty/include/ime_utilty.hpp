#pragma once

#include<string>

#include"opencv2/opencv.hpp"

namespace rw {
	using Point = std::pair<double, double>;

    struct DetectionRectangleInfo
    {
    public:
        Point leftTop{};
        Point rightTop{};
        Point leftBottom{};
        Point rightBottom{};
    public:
        int center_x{ -1 };
        int center_y{ -1 };
    public:
        int width{ -1 };
        int height{ -1 };
    public:
        long area{ -1 };
    public:
        size_t classId{ 0 };
        double score{ -1 };
    };

	struct ImagePainter
	{
        enum class ShapeType {
            Rectangle,
            Circle
        };

        struct PainterConfig
        {
            ShapeType shapeType{ ShapeType ::Rectangle};
            int fontSize=5;
            int fontThickness = 1;
            int thickness=2;
            std::string text;
            cv::Scalar color{ 0, 0, 255 };
			cv::Scalar textColor{ 0, 255, 0 };
        };

        static cv::Mat drawShapes(
            const cv::Mat& image,
            const std::vector<DetectionRectangleInfo>& rectInfo,
            PainterConfig config
        );

        static void drawShapesOnSourceImg(
			cv::Mat& image,
            const std::vector<DetectionRectangleInfo>& rectInfo,
            PainterConfig config
        );


        static cv::Mat drawShapes(
            const cv::Mat& image,
            const DetectionRectangleInfo& rectInfo,
            PainterConfig config
        );



        static void drawShapesOnSourceImg(
            cv::Mat& image, 
            const DetectionRectangleInfo& rectInfo,
            PainterConfig config
        );
	};

	
}
