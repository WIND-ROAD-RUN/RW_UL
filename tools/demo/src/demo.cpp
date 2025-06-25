#include "demo.h"

#include"rqw_ImagePainter.h"

QImage cvMatToQImage(const cv::Mat& mat)
{
	return rw::rqw::cvMatToQImage(mat);
}

demo::demo(QWidget* parent)
	: QMainWindow(parent)
	, ui(new Ui::demoClass())
{
	ui->setupUi(this);

	auto camerList = rw::rqw::CheckCameraList(rw::rqw::CameraProvider::DS);
	if (camerList.empty())
	{
		qDebug() << "No camera found!";
		return;
	}
	m_cameraThread.initCamera(camerList[0], rw::rqw::CameraObjectTrigger::Software);
	QObject::connect(&m_cameraThread, &rw::rqw::CameraPassiveThread::frameCaptured, this, &demo::displayImg);
	m_cameraThread.startMonitor();
	m_cameraThread.setFrameRate(5);

	rw::ModelEngineConfig config;
	config.modelPath = R"(C:\Users\zfkj\Desktop\best.engine)";
	config.nms_threshold = 0.1;
	config.conf_threshold = 0.1;
	//engine=rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::Yolov11_Seg, rw::ModelEngineDeployType::TensorRT);
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
	//auto result = engine->processImg(frame);
	auto QImage = cvMatToQImage(frame);
	//rw::rqw::ImagePainter::drawShapesOnSourceImg(QImage, result);
	ui->label->setPixmap(QPixmap::fromImage(QImage).scaled(ui->label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}