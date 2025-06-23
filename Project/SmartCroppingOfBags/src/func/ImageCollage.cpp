#include"ImageCollage.hpp"
#include"rqw_HistoricalElementManager.hpp"

#include<QPainter>

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
            //std::cout << result.value().attribute["location"]<<std::endl;
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
	if (images.empty()) {
		return cv::Mat(); // 如果列表为空，返回空的 Mat
	}

	// 计算拼接后图像的总高度和最大宽度
	int totalHeight = 0;
	int maxWidth = 0;
	for (const auto& img : images) {
		totalHeight += img.rows; // 累加每张图片的高度
		maxWidth = std::max(maxWidth, img.cols); // 找到最大的宽度
	}

	// 创建拼接后的图像，初始化为黑色背景
	cv::Mat result(totalHeight, maxWidth, images[0].type(), cv::Scalar(0, 0, 0));

	// 按顺序将每张图片拷贝到拼接图像中
	int currentY = 0;
	for (const auto& img : images) {
		cv::Rect roi(0, currentY, img.cols, img.rows); // 定义拷贝区域
		img.copyTo(result(roi)); // 拷贝图像到结果图像的指定区域
		currentY += img.rows; // 更新当前 Y 坐标
	}

	return result; // 返回拼接后的图像
}

QImage ImageCollage::verticalConcat(const QVector<QImage>& images)
{
    if (images.isEmpty()) return QImage();

    int width = images[0].width();
    int totalHeight = 0;

    // 检查所有图片宽度是否一致，并计算总高度
    for (const auto& img : images) {
        if (img.isNull() || img.width() != width) {
            // 尺寸不一致或图片无效，返回空
            return QImage();
        }
        totalHeight += img.height();
    }

    // 创建目标QImage，使用ARGB32格式以兼容性最佳
    QImage result(width, totalHeight, QImage::Format_ARGB32);
    result.fill(Qt::transparent);

    QPainter painter(&result);
    int currentY = 0;
    for (const auto& img : images) {
        painter.drawImage(0, currentY, img);
        currentY += img.height();
    }
    painter.end();

    return result;
}

size_t ImageCollage::size()
{
    return _imageManager->size();
}

