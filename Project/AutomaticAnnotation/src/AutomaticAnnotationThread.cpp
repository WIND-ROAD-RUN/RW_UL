#include"AutomaticAnnotationThread.h"

#include"ime_ModelEngineFactory.h"
#include"rqw_ImagePainter.h"
#include<QMessageBox>

AutomaticAnnotationThread::AutomaticAnnotationThread(const QVector<QString>& imagePaths, QObject* parent)
    : QThread(parent), m_imagePaths(imagePaths) {

}

void AutomaticAnnotationThread::run()
{
    auto engine = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::yolov11_obb, rw::ModelEngineDeployType::TensorRT);
    if (engine == nullptr)
    {
        return;
    }

    for (const QString& path : m_imagePaths) {
		auto mat = cv::imread(path.toStdString());
		if (mat.empty()) {
			qDebug() << "Failed to load image:" << path;
			continue;
		}
        auto result = engine->processImg(mat);

        auto image=rw::rqw::ImagePainter::cvMatToQImage(mat);

		rw::rqw::ImagePainter::drawShapesOnSourceImg(image, result);

		QPixmap pixmap = QPixmap::fromImage(image);

        emit imageProcessed(path, pixmap);
    }

}
