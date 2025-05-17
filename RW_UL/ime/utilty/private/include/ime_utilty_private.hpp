#pragma once

#include"opencv2/opencv.hpp"

#include <string>
#include <vector>

namespace rw
{
	std::vector<int> nmsWithKeepClass(
		const std::vector<cv::Rect>& boxes,
		const std::vector<int>& class_ids,
		const std::vector<float>& confidences,
		float conf_threshold,
		float nms_threshold,
		const std::vector<size_t>& need_keep_classids);
	
}
