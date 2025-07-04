#include <QPainter>
#include <QtWidgets/QApplication>
#include"PicturesPainter.h"

int main(int argc, char* argv[])
{
	QApplication a(argc, argv);
	// 1. 创建一个配置 vector
	std::vector<RectangeConfig> configs;

	// 可以根据需要往 configs 填入内容，例如
	configs.push_back(RectangeConfig{ 1, QColor(Qt::red), "黑疤", "黑疤" });
	configs.push_back(RectangeConfig{ 2, QColor(Qt::green), "疏档", "疏档" });
    configs.push_back(RectangeConfig{ 2, QColor(Qt::blue), "框", "1" });

	PicturesPainter p;
	p.setRectangleConfigs(configs);

	// 2. 读取图片
	QString imagePath = R"(C:\Users\zzw\Desktop\123.jpeg)"; // 绝对路径或相对路径
	QImage image(imagePath);
	if (image.isNull()) {
		qDebug() << "加载图片失败:" << imagePath;
		return -1;
	}

	p.setImage(image); 
    //p.setAspectRatio(300, 200);

    std::vector<PicturesPainter::PainterRectangleInfo> rectangles;

    // 框1
    rectangles.push_back({
        {100, 100},   // leftTop
        {200, 100},   // rightTop
        {100, 200},   // leftBottom
        {200, 200},   // rightBottom
        150,          // center_x
        150,          // center_y
        100,          // width
        100,          // height
        10000,        // area
        1,            // classId
        0.92          // score
        });

    // 框2
    rectangles.push_back({
        {50, 50},
        {120, 50},
        {50, 120},
        {120, 120},
        85,    // center_x
        85,    // center_y
        70,    // width
        70,    // height
        4900,  // area
        2,
        0.87
        });

    // 框3
    rectangles.push_back({
        {220, 80},
        {320, 80},
        {220, 180},
        {320, 180},
        270,    // center_x
        130,    // center_y
        100,    // width
        100,    // height
        10000,  // area
        1,
        0.75
        });

    p.setDrawnRectangles(rectangles);
	auto result=p.exec();
    if (result==QDialog::Accepted)
    {

       auto rets= p.getRectangleConfigs();
    }



	return a.exec();
}
