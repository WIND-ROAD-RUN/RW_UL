#include "demo.h"

#include"rqw_ImagePainter.h"

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

demo::demo(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::demoClass())
{
	ui->setupUi(this);

	auto camerList=rw::rqw::CheckCameraList();
	if (camerList.empty())
	{
		qDebug() << "No camera found!";
		return;
	}
	m_cameraThread.initCamera(camerList[0], rw::rqw::CameraObjectTrigger::Software,0);
	QObject::connect(&m_cameraThread, &rw::rqw::CameraPassiveThread::frameCaptured, this, &demo::displayImg);
	m_cameraThread.startMonitor();
	m_cameraThread.setFrameRate(5);

	rw::ModelEngineConfig config;
	config.modelPath = R"(C:\Users\zfkj\Desktop\best.engine)";
	config.nms_threshold = 0.1;
	config.conf_threshold = 0.1;
	engine=rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::Yolov11_Seg, rw::ModelEngineDeployType::TensorRT);

}

demo::~demo()
{
	delete ui;
}

void demo::displayImg(cv::Mat frame)
{
	if (frame.empty())
	{
		return;
	}
	auto result = engine->processImg(frame);
	auto QImage = cvMatToQImage(frame);
	rw::rqw::ImagePainter::drawShapesOnSourceImg(QImage, result);
	ui->label->setPixmap(QPixmap::fromImage(QImage).scaled(ui->label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
