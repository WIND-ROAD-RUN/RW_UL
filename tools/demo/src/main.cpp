#include <QPainter>
#include <QtWidgets/QApplication>

#include"PictureViewerThumbnails.h"
#include "FullKeyBoard.h"

#include"LicenseValidation.h"

#include"rqw_ImagePainter.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);

	// 创建一个 640x640 的黄色 QImage
	//QImage image(640, 640, QImage::Format_RGB32);
	//image.fill(Qt::yellow); // 填充为黄色

	//rw::rqw::ImagePainter::PainterConfig config;
	//config.text = "sadawdawdadwd";
	//config.textLocate = rw::rqw::ImagePainter::PainterConfig::TextLocate::RightBottomOut;
	//rw::DetectionRectangleInfo det;
	//det.leftBottom = { 100, 500 };
	//det.leftTop = { 100, 100 };
	//det.rightBottom = { 500, 500 };
	//det.rightTop = { 500, 100 };
	//det.center_x = 150;
	//det.center_y = 150;
	//rw::rqw::ImagePainter::drawShapesOnSourceImg(image, det, config);
	//QLabel label;
	//label.setPixmap(QPixmap::fromImage(image));
	//label.show();

	LicenseValidation l;
	l.show();

	return a.exec();
}
