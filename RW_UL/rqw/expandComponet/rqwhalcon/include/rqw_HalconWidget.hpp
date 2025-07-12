#pragma once

#include <QWidget>
#include"opencv2/opencv.hpp"

namespace HalconCpp
{
	class HTuple;
	class HObject;
    class HImage;
}


namespace rw {
	namespace rqw {

        class HalconWidget : public QWidget
        {
            Q_OBJECT
        public:
            //TODO:appendHObject接口
			//TODO:clearHObject接口
        public:
            explicit HalconWidget(QWidget* parent = nullptr);
            ~HalconWidget() override;
        public:
            void setImage(const HalconCpp::HImage& image);
            void setImage(const QImage& image);
            void setImage(const cv::Mat& mat);
        public:
            HalconCpp::HTuple* _halconWindowHandle{ nullptr };
            HalconCpp::HImage* _image{ nullptr };
        private:
            void initializeHalconWindow();
            void closeHalconWindow();
            void displayImg();
        public:
            void wheelEvent(QWheelEvent* event) override; 
        protected:
            void showEvent(QShowEvent* event) override;
            void resizeEvent(QResizeEvent* event) override;
        protected:
            void mousePressEvent(QMouseEvent* event) override;
            void mouseMoveEvent(QMouseEvent* event) override;
            void mouseReleaseEvent(QMouseEvent* event) override;
        private:
            bool _isDragging{ false }; 
            QPoint _lastMousePos;
            bool _isDrawingRect{ false };
        public slots:
            void drawRect();
        };

	}
}