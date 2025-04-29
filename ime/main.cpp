#include"opencv2/opencv.hpp"

int main() {
	auto a=cv::imread(R"(C:\Users\zfkj\Desktop\1.png)", cv::IMREAD_COLOR);
	cv::imshow("a",a);
	cv::waitKey(0);

	return 0;
}