#include"stdafx.h"

#include"ButtonUtilty.h"

QImage cvMatToQImage(const cv::Mat& mat)
{
	QImage result;
	if (mat.type() == CV_8UC1) {
		result = QImage(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_Grayscale8);
	}
	else if (mat.type() == CV_8UC3) {
		result = QImage(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_RGB888).rgbSwapped();
	}
	else if (mat.type() == CV_8UC4) {
		result = QImage(mat.data, mat.cols, mat.rows, mat.step[0], QImage::Format_RGBA8888);
	}
	else {
		result = QImage();
	}

	return result;
}

QPixmap cvMatToQPixmap(const cv::Mat& mat)
{
	QImage image = cvMatToQImage(mat);
	if (image.isNull()) {
		return QPixmap();
	}
	else {
		return QPixmap::fromImage(image);
	}
}