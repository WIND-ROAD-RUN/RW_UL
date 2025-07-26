#include"imgPro_ImageProcess_t.hpp"

TEST_F(ImageProcessTest, ImageProcess)
{
	int argc = 0;
	char* argv[] = { nullptr };
	QApplication app(argc, argv);

	cv::Mat image = cv::imread(R"(C:\Users\rw\Desktop\temp\niukou.png)");
	(*imgProcess)(image);
	auto maskImg = imgProcess->getMaskImg(image);

	QLabel label;
	label.setPixmap(QPixmap::fromImage(maskImg));
	label.show();

	app.exec();
}