#include"ImageCollage.hpp"
#include"rqw_HistoricalElementManager.hpp"

ImageCollage::ImageCollage()
{
}

ImageCollage::~ImageCollage()
{
}

void ImageCollage::iniCache(size_t capacity)
{
	_imageManager = std::make_unique<rw::rqw::HistoricalElementManager<Time, cv::Mat>>(capacity);
}

bool ImageCollage::pushImage(const rw::rqw::ElementInfo<cv::Mat>& image, const Time& time)
{
	 auto getResult=_imageManager->getElement(time);
	if (getResult.has_value())
	{
		return false;
	}
	_imageManager->insertElement(time, image);
	return true;
}


std::optional<rw::rqw::ElementInfo<cv::Mat>> ImageCollage::getImage(const Time& time)
{
	return _imageManager->getElement(time);
}

ImageCollage::CollageImage ImageCollage::getCollageImage(const std::vector<Time>& times, bool& hasNull)
{
    hasNull = false;
    std::vector<cv::Mat> images;
    std::vector<Time> timesResult;

    
    for (const auto& time : times)
    {
        auto result = _imageManager->getElement(time);
        if (result.has_value())
        {
            images.push_back(result.value().element);
            timesResult.emplace_back(time);
        }
        else
        {
            hasNull = true; 
        }
    }

    
    if (images.empty())
    {
        return CollageImage(cv::Mat());
    }

    
    int totalHeight = 0;
    int maxWidth = 0;
    for (const auto& img : images)
    {
        totalHeight += img.rows; 
        maxWidth = std::max(maxWidth, img.cols); 
    }

    
    cv::Mat collageImage(totalHeight, maxWidth, images[0].type(), cv::Scalar(0, 0, 0)); // 初始化为黑色背景

    
    int currentY = 0;
    for (const auto& img : images)
    {
        img.copyTo(collageImage(cv::Rect(0, currentY, img.cols, img.rows))); // 将图片拷贝到拼接图像中
        currentY += img.rows; 
    }


    CollageImage result(collageImage);
    result.times = timesResult;
    
    return result;

}

ImageCollage::CollageImage ImageCollage::getCollageImage(const std::vector<Time>& times)
{
    bool hasNull{false};
    return getCollageImage(times, hasNull);
}

cv::Mat ImageCollage::verticalConcat(const std::vector<cv::Mat>& images)
{
    if (images.empty()) return cv::Mat();

    // 检查所有图片宽度是否一致
    int width = images[0].cols;
    for (const auto& img : images) {
        if (img.empty() || img.cols != width) {
            // 尺寸不一致，返回空
            return cv::Mat();
        }
    }

    // 计算总高度
    int totalHeight = 0;
    for (const auto& img : images) {
        totalHeight += img.rows;
    }

    // 创建输出图像
    cv::Mat result(totalHeight, width, images[0].type());
    int currentY = 0;
    for (const auto& img : images) {
        img.copyTo(result.rowRange(currentY, currentY + img.rows));
        currentY += img.rows;
    }
    return result;
}

