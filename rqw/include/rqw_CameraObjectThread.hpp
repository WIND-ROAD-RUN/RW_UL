#pragma once

#include"rqw_CameraObjectCore.hpp"

#include<opencv2/opencv.hpp>

#include<QThread>


namespace rw {
    namespace rqw {
        class CameraPassiveObject;

        class CameraPassiveThread : public QThread
        {
            Q_OBJECT
        public:
            explicit CameraPassiveThread(QObject* parent = nullptr);
            ~CameraPassiveThread() override;

            void initCamera(const rw::rqw::CameraMetaData& cameraMetaData, rw::rqw::CameraObjectTrigger triggerMode);
            void startMonitor();
            void stopMonitor();

        public:
            //TODO: Add more functions
            void setExposureTime(size_t value) const;
            void setGain(size_t value) const;
            void setIOTime(size_t value) const;
            void setTriggerMode(CameraObjectTrigger mode) const;
            void setTriggerLine(size_t lineIndex)const;
        public:
            //TODO: Add more functions
            [[nodiscard]] size_t getExposureTime() const;
            [[nodiscard]] size_t getGain() const;
            [[nodiscard]] size_t getIOTime() const;
            [[nodiscard]] CameraObjectTrigger getMonitorMode() const;
            [[nodiscard]] size_t getTriggerLine() const;

        signals:
            void frameCaptured(cv::Mat frame);
            void frameCapturedWithMetaData(cv::Mat frame, rw::rqw::CameraMetaData cameraMetaData);
            void frameCapturedWithoutArgs();

        protected:
            void run() override;

        private:
            CameraPassiveObject* _cameraObject;
        private slots:
            void onFrameCaptured(cv::Mat frame);
            void onFrameCapturedWithMetaData(cv::Mat frame, rw::rqw::CameraMetaData cameraMetaData);
            void onFrameCapturedWithoutArgs();
        };


    } // namespace rqw

} // namespace rw
