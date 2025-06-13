#pragma once

#include"opencv2/opencv.hpp"
#include<memory>

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
private:
	std::vector<rw::rqw::HistoricalElementManager<std::chrono::system_clock::time_point, cv::Mat>> imageManager;



};