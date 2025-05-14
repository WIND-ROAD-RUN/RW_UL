#include"ime_utilty.hpp"

namespace rw
{
	cv::Scalar ImagePainter::toScalar(BasicColor color)
	{
		switch (color) {
		case BasicColor::Red:        return cv::Scalar(0, 0, 255);   // BGR: Red
		case BasicColor::Green:      return cv::Scalar(0, 255, 0);   // BGR: Green
		case BasicColor::Blue:       return cv::Scalar(255, 0, 0);   // BGR: Blue
		case BasicColor::Yellow:     return cv::Scalar(0, 255, 255); // BGR: Yellow
		case BasicColor::Cyan:       return cv::Scalar(255, 255, 0); // BGR: Cyan
		case BasicColor::Magenta:    return cv::Scalar(255, 0, 255); // BGR: Magenta
		case BasicColor::White:      return cv::Scalar(255, 255, 255); // BGR: White
		case BasicColor::Black:      return cv::Scalar(0, 0, 0);     // BGR: Black
		case BasicColor::Orange:     return cv::Scalar(0, 165, 255); // BGR: Orange
		case BasicColor::LightBlue:  return cv::Scalar(255, 182, 193); // BGR: Light Blue
		case BasicColor::Gray:       return cv::Scalar(128, 128, 128); // BGR: Gray
		case BasicColor::Purple:     return cv::Scalar(128, 0, 128);   // BGR: Purple
		case BasicColor::Brown:      return cv::Scalar(42, 42, 165);   // BGR: Brown
		case BasicColor::LightBrown: return cv::Scalar(181, 229, 255); // BGR: Light Brown
		default:                     return cv::Scalar(0, 0, 0);     // Default to Black
		}
	}

	void ImagePainter::drawTextOnImage(cv::Mat& mat, const std::vector<std::string>& texts, const std::vector<PainterConfig>& colorList, double proportion)
	{
	
		if (texts.empty() || proportion <= 0.0 || proportion > 1.0 || mat.empty()) {
			return; 
		}

		int imageHeight = mat.rows;
		int fontSize = static_cast<int>(imageHeight * proportion); 

		int x = 10; 
		int y = fontSize; 

		for (size_t i = 0; i < texts.size(); ++i) {
			cv::Scalar textColor = (i < colorList.size()) ? colorList[i].textColor : colorList.back().textColor;

			cv::putText(
				mat,
				texts[i], 
				cv::Point(x, y), 
				cv::FONT_HERSHEY_SIMPLEX, 
				proportion, 
				textColor, 
				colorList[i].fontThickness, 
				cv::LINE_AA 
			);

			y += fontSize + 5; 
		}
	}

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

	void ImagePainter::drawShapesOnSourceImg(cv::Mat& image, const std::vector<std::vector<size_t>> index,
		const std::vector<DetectionRectangleInfo>& rectInfo, PainterConfig config)
	{
		for (const auto& classId : index) {
			for (const auto& item : classId)
			{
				config.text = std::to_string(rectInfo[item].classId);
				config.fontSize = 10;
				drawShapesOnSourceImg(image, rectInfo[item], config);
			}
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

	std::vector<int> nmsWithKeepClass(
		const std::vector<cv::Rect>& boxes,
		const std::vector<int>& class_ids,
		const std::vector<float>& confidences,
		float conf_threshold,
		float nms_threshold,
		const std::vector<size_t>& need_keep_classids)
	{
		std::vector<int> nms_result;
		std::set<size_t> keep_set(need_keep_classids.begin(), need_keep_classids.end());

		if (need_keep_classids.empty()) {
			// 全部一起NMS
			cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
			return nms_result;
		}

		// 先处理不在need_keep_classids中的框
		std::vector<cv::Rect> other_boxes;
		std::vector<float> other_confidences;
		std::vector<int> other_indices;
		for (int i = 0; i < class_ids.size(); ++i) {
			if (keep_set.count(class_ids[i]) == 0) {
				other_boxes.push_back(boxes[i]);
				other_confidences.push_back(confidences[i]);
				other_indices.push_back(i);
			}
		}
		std::vector<int> other_nms;
		if (!other_boxes.empty())
			cv::dnn::NMSBoxes(other_boxes, other_confidences, conf_threshold, nms_threshold, other_nms);
		for (int idx : other_nms) {
			nms_result.push_back(other_indices[idx]);
		}

		// 对每个需要单独NMS的classid分别处理
		for (size_t cid : keep_set) {
			std::vector<cv::Rect> class_boxes;
			std::vector<float> class_confidences;
			std::vector<int> class_indices;
			for (int i = 0; i < class_ids.size(); ++i) {
				if (class_ids[i] == static_cast<int>(cid)) {
					class_boxes.push_back(boxes[i]);
					class_confidences.push_back(confidences[i]);
					class_indices.push_back(i);
				}
			}
			std::vector<int> class_nms;
			if (!class_boxes.empty())
				cv::dnn::NMSBoxes(class_boxes, class_confidences, conf_threshold, nms_threshold, class_nms);
			for (int idx : class_nms) {
				nms_result.push_back(class_indices[idx]);
			}
		}

		return nms_result;
	}

	std::vector<rw::DetectionRectangleInfo>::const_iterator DetectionRectangleInfo::getMaxAreaRectangleIterator(
		const std::vector<rw::DetectionRectangleInfo>& bodyIndexVector)
	{
		if (bodyIndexVector.empty()) {
			return bodyIndexVector.end(); 
		}

		return std::max_element(
			bodyIndexVector.begin(),
			bodyIndexVector.end(),
			[](const rw::DetectionRectangleInfo& a, const rw::DetectionRectangleInfo& b) {
				return a.area < b.area; 
			}
		);
	}

	std::vector<rw::DetectionRectangleInfo>::const_iterator DetectionRectangleInfo::getMaxAreaRectangleIterator(const std::vector<rw::DetectionRectangleInfo>& bodyIndexVector,const std::vector<size_t> & index)
	{
		if (bodyIndexVector.empty()) {
			return bodyIndexVector.end();
		}
		if (index.empty())
		{
			return bodyIndexVector.end();
		}
		auto currentArea = bodyIndexVector[index[0]].area;
		int maxIndex = index[0];
		for (int i = 0;i< index.size();i++)
		{
			if (bodyIndexVector[index[i]].area > currentArea)
			{
				currentArea = bodyIndexVector[index[i]].area;
				maxIndex = index[i];
			}
		}

		return bodyIndexVector.begin() + maxIndex;

	}


}
