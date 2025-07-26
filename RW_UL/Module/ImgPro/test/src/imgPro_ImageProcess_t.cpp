#include"imgPro_ImageProcess_t.hpp"

#include "rqw_ImgConvert.hpp"
#include "ime_utilty.hpp"


TEST_F(ImageProcessTest, ImageProcess)
{
	int argc = 0;
	char* argv[] = { nullptr };
	QApplication app(argc, argv);

	cv::Mat image = cv::imread(R"(C:\Users\rw\Desktop\temp\2025-06-17_11-10-55-776.jpg)");
	(*imgProcess)(image);
	auto maskImg = imgProcess->getMaskImg(image);

	//rw::imgPro::ConfigDrawMask cfg;
	//cfg.color = rw::rqw::RQWColor::Red;
	//for (const auto & item: imgProcess->getProcessResult())
	//{
	//	rw::ImagePainter::PainterConfig config;
	//	
	//	if (item.classId==0)
	//	{
	//		cfg.color = rw::rqw::RQWColor::Green;
	//	}
	//	if (item.classId == 1)
	//	{
	//		cfg.color = rw::rqw::RQWColor::Red;
	//	}
	//	if (item.classId == 2)
	//	{
	//		cfg.color = rw::rqw::RQWColor::Blue;
	//	}

	//	//rw::ImagePainter::drawMaskOnSourceImg(image,item, config);
	//	rw::imgPro::ImagePainter::drawMaskOnSourceImg(image, item, cfg);
	//}
	//auto maskImg = rw::CvMatToQImage(image);



	//rw::imgPro::ConfigDrawMask cfg;
	//cfg.color = rw::rqw::RQWColor::Red;
	//auto maskImg = rw::CvMatToQImage(image);
	//for (const auto& item : imgProcess->getProcessResult())
	//{
	//	rw::ImagePainter::PainterConfig config;

	//	if (item.classId == 0)
	//	{
	//		cfg.color = rw::rqw::RQWColor::Green;
	//	}
	//	if (item.classId == 1)
	//	{
	//		cfg.color = rw::rqw::RQWColor::Red;
	//	}
	//	if (item.classId == 2)
	//	{
	//		cfg.color = rw::rqw::RQWColor::Blue;
	//	}

	//	//rw::ImagePainter::drawMaskOnSourceImg(image,item, config);
	//	rw::imgPro::ImagePainter::drawMaskOnSourceImg(maskImg, item, cfg);
	//}


	//rw::imgPro::ConfigDrawRect cfg;
	//cfg.rectColor = rw::rqw::RQWColor::Red;
	//cfg.isRegion = true;
	//cfg.hasFrame = false;
	//cfg.alpha = 0.5;
	//auto maskImg = rw::CvMatToQImage(image);
	//for (const auto& item : imgProcess->getProcessResult())
	//{
	//	rw::ImagePainter::PainterConfig config;

	//	if (item.classId == 0)
	//	{
	//		cfg.rectColor = rw::rqw::RQWColor::Green;
	//	}
	//	if (item.classId == 1)
	//	{
	//		cfg.rectColor = rw::rqw::RQWColor::Red;
	//	}
	//	if (item.classId == 2)
	//	{
	//		cfg.rectColor = rw::rqw::RQWColor::Blue;
	//	}

	//	//rw::ImagePainter::drawMaskOnSourceImg(image,item, config);
	//	rw::imgPro::ImagePainter::drawShapesOnSourceImg(maskImg, item, cfg);
	//}



	QLabel label;
	label.setPixmap(QPixmap::fromImage(maskImg));
	label.show();

	app.exec();
}