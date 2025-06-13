#pragma once

#include"opencv2/opencv.hpp"
#include<memory>

#include "rqw_HistoricalElementManager.hpp"
#include"Utilty.hpp"

namespace std {
	template <>
	struct hash<std::chrono::system_clock::time_point> {
		size_t operator()(const std::chrono::system_clock::time_point& timePoint) const noexcept {
			return std::hash<int64_t>()(std::chrono::duration_cast<std::chrono::microseconds>(timePoint.time_since_epoch()).count());
		}
	};
}

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
};


