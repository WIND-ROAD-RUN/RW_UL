#include <QDebug>
#include"rqw_CameraDisplayLabel.hpp"

#include <QMouseEvent>
#include <QDebug>
#include"opencv2/opencv.hpp"
#include<QThread>

namespace rw
{
    namespace rqw
    {
        CameraDisplayLabel::CameraDisplayLabel(QWidget* parent) : QLabel(parent)
        {
            auto cameraList=CheckCameraList();
            if (cameraList.size() < 0)
            {
                return;
            }

            CameraMetaData & cameraMetaData = cameraList[0];
            _cameraPassiveThread = new CameraPassiveThread();
            _cameraPassiveThread->initCamera(cameraMetaData, CameraObjectTrigger::Software);

            //多线程用Qt::QueuedConnection
            connect(_cameraPassiveThread, &CameraPassiveThread::frameCapturedWithMetaData, this, &CameraDisplayLabel::onFrameCapturedWithMetaData, Qt::QueuedConnection);
            _cameraPassiveThread->startMonitor();
            _cameraPassiveThread->setExposureTime(18000);
            _cameraPassiveThread->setGain(10);
        }

        CameraDisplayLabel::~CameraDisplayLabel()
        {
            _cameraPassiveThread->stopMonitor();

            delete _cameraPassiveThread;
        }


        void CameraDisplayLabel::onFrameCapturedWithMetaData(cv::Mat frame, CameraMetaData cameraMetaData)
        {
            static int count = 0;
            QString text = QString("Frame captured: %1").arg(count++);
            QString framInfo = QString("Frame Info: %1").arg(cameraMetaData.ip);
            QString framSize = QString("Frame Size: %1x%2").arg(frame.cols).arg(frame.rows);
            text += QString("\n") + framInfo + QString("\n") + framSize;
            setText(text);

            cv::Mat rgbFrame;
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
            QImage img = QImage((const unsigned char*)(rgbFrame.data), rgbFrame.cols, rgbFrame.rows, QImage::Format_RGB888);
            setPixmap(QPixmap::fromImage(img));
        }
    } // namespace rsw
} // namespace rw