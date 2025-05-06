#pragma once

#include<string>

#include"opencv2/opencv.hpp"

namespace rw {
	using Point = std::pair<int, int>;

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
	public:
		static std::vector<rw::DetectionRectangleInfo>::const_iterator getMaxAreaRectangleIterator(
			const std::vector<rw::DetectionRectangleInfo>& bodyIndexVector);
		static std::vector<rw::DetectionRectangleInfo>::const_iterator getMaxAreaRectangleIterator(
			const std::vector<rw::DetectionRectangleInfo>& bodyIndexVector, const std::vector<size_t>& index);
	};

	using PointScale = std::pair<double, double>;
	struct DetectionRectangleScaleInfo
	{
	public:
		PointScale leftTop{};
		PointScale rightTop{};
		PointScale leftBottom{};
		PointScale rightBottom{};
	public:
		double center_x{ -1 };
		double center_y{ -1 };
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

		enum class BasicColor {
			Red,
			Green,
			Blue,
			Yellow,
			Cyan,
			Magenta,
			White,
			Black,
			Orange,
			LightBlue, 
			Gray,      
			Purple,    
			Brown,    
			LightBrown 
		};


		static cv::Scalar toScalar(BasicColor color);

		struct PainterConfig
		{
			ShapeType shapeType{ ShapeType::Rectangle };
			int fontSize = 5;
			int fontThickness = 1;
			int thickness = 2;
			std::string text;
			cv::Scalar color{ 0, 0, 255 };
			cv::Scalar textColor{ 0, 255, 0 };
		};

		static void drawTextOnImage(cv::Mat& mat, const std::vector<std::string>& texts, const std::vector<PainterConfig>& colorList, double proportion = 0.8);

		static cv::Mat drawShapes(
			const cv::Mat& image,
			const std::vector<DetectionRectangleInfo>& rectInfo,
			PainterConfig config
		);

		static void drawShapesOnSourceImg(
			cv::Mat& image,
			const std::vector<DetectionRectangleInfo>& rectInfo,
			PainterConfig config = {}
		);

		static void drawShapesOnSourceImg(cv::Mat& image, const std::vector<std::vector<size_t>> index, const std::vector<DetectionRectangleInfo>& rectInfo,
			PainterConfig config = {});

		static cv::Mat drawShapes(
			const cv::Mat& image,
			const DetectionRectangleInfo& rectInfo,
			PainterConfig config = {}
		);

		static void drawShapesOnSourceImg(
			cv::Mat& image,
			const DetectionRectangleInfo& rectInfo,
			PainterConfig config = {}
		);

		static  void drawVerticalLine(cv::Mat& image, int position, const ImagePainter::PainterConfig& config);
		static void drawHorizontalLine(cv::Mat& image, int position, const ImagePainter::PainterConfig& config);
	};

}
