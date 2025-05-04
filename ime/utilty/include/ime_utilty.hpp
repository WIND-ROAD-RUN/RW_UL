#pragma once

#include<string>

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
		double center_x{-1};
		double center_y{-1};
	public:
		double width{-1};
		double height{-1};
	public:
		double area{-1};
	public:
		size_t classId{0};
		double score{-1};
	};
}
