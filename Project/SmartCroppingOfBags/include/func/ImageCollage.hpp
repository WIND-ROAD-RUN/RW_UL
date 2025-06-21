#pragma once

#include"opencv2/opencv.hpp"
#include<memory>

#include "rqw_HistoricalElementManager.hpp"
#include"Utilty.hpp"

//namespace std {
//	template <>
//	struct hash<std::chrono::system_clock::time_point> {
//		size_t operator()(const std::chrono::system_clock::time_point& timePoint) const noexcept {
//			// 将 timePoint 转换为精确到纳秒的字符串
//			std::ostringstream oss;
//			std::time_t timeT = std::chrono::system_clock::to_time_t(timePoint);
//			auto duration = timePoint.time_since_epoch();
//			auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration) % 1000000000;
//
//			oss << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S")
//				<< '.' << std::setfill('0') << std::setw(9) << nanoseconds.count();
//
//			// 对字符串进行哈希处理
//			return std::hash<std::string>()(oss.str());
//		}
//	};
//}

namespace rw
{
	namespace rqw
	{
		template <typename KeyType, typename ValueType>
		class HistoricalElementManager;
	}
}



class ImageCollage
{
public:
	struct CollageImage
	{
	public:
		cv::Mat mat;
		std::vector<Time> times;
	public:
		CollageImage() = default;
		CollageImage(const cv::Mat& img) : mat(img) {}
	};
private:
	std::unique_ptr<rw::rqw::HistoricalElementManager<Time,cv::Mat>> _imageManager;
public:
	ImageCollage();
	~ImageCollage();
public:
	void iniCache(size_t capacity);
public:
	bool pushImage(const rw::rqw::ElementInfo<cv::Mat> & image,const Time &time);
	std::optional<rw::rqw::ElementInfo<cv::Mat>> getImage(const Time& time);
	CollageImage getCollageImage(const std::vector<Time> & times,bool & hasNull);
	CollageImage getCollageImage(const std::vector<Time>& times);

	static cv::Mat verticalConcat(const std::vector<cv::Mat>& images);
	static QImage verticalConcat(const QVector<QImage>& images);
	
};


