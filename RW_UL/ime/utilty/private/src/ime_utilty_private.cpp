#include"ime_utilty_private.hpp"

namespace rw
{
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

		if (boxes.empty()) return nms_result;

		if (need_keep_classids.empty()) {
			// 所有类别分别单独NMS
			std::map<int, std::vector<int>> class_to_indices;
			for (int i = 0; i < class_ids.size(); ++i) {
				class_to_indices[class_ids[i]].push_back(i);
			}
			for (const auto& kv : class_to_indices) {
				std::vector<cv::Rect> class_boxes;
				std::vector<float> class_confidences;
				std::vector<int> class_nms;
				for (int idx : kv.second) {
					class_boxes.push_back(boxes[idx]);
					class_confidences.push_back(confidences[idx]);
				}
				if (!class_boxes.empty())
					cv::dnn::NMSBoxes(class_boxes, class_confidences, conf_threshold, nms_threshold, class_nms);
				for (int nms_idx : class_nms) {
					nms_result.push_back(kv.second[nms_idx]);
				}
			}
			return nms_result;
		}

		// 1. 需要一起NMS的类别
		std::vector<cv::Rect> keep_boxes;
		std::vector<float> keep_confidences;
		std::vector<int> keep_indices;
		// 2. 其他类别分组
		std::map<int, std::vector<int>> class_to_indices;
		for (int i = 0; i < class_ids.size(); ++i) {
			if (keep_set.count(class_ids[i])) {
				keep_boxes.push_back(boxes[i]);
				keep_confidences.push_back(confidences[i]);
				keep_indices.push_back(i);
			}
			else {
				class_to_indices[class_ids[i]].push_back(i);
			}
		}
		// 1. 对 need_keep_classids 里的框一起NMS
		std::vector<int> keep_nms;
		if (!keep_boxes.empty())
			cv::dnn::NMSBoxes(keep_boxes, keep_confidences, conf_threshold, nms_threshold, keep_nms);
		for (int nms_idx : keep_nms) {
			nms_result.push_back(keep_indices[nms_idx]);
		}
		// 2. 其他类别分别单独NMS
		for (const auto& kv : class_to_indices) {
			std::vector<cv::Rect> class_boxes;
			std::vector<float> class_confidences;
			std::vector<int> class_nms;
			for (int idx : kv.second) {
				class_boxes.push_back(boxes[idx]);
				class_confidences.push_back(confidences[idx]);
			}
			if (!class_boxes.empty())
				cv::dnn::NMSBoxes(class_boxes, class_confidences, conf_threshold, nms_threshold, class_nms);
			for (int nms_idx : class_nms) {
				nms_result.push_back(kv.second[nms_idx]);
			}
		}
		return nms_result;
	}
}