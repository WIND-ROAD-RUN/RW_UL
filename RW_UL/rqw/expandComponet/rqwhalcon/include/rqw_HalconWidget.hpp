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
        class HalconWidget;

        struct HalconWidgetDisObject
        {
            friend HalconWidget;
        public:
	        explicit HalconWidgetDisObject(HalconCpp::HObject* obj);
            explicit HalconWidgetDisObject(const HalconCpp::HImage& image);
            explicit HalconWidgetDisObject(const cv::Mat& mat);
            explicit HalconWidgetDisObject(const QImage& image);

            ~HalconWidgetDisObject();
        public:
            HalconWidgetDisObject(const HalconWidgetDisObject& other);
            HalconWidgetDisObject(HalconWidgetDisObject&& other) noexcept;
            HalconWidgetDisObject& operator=(const HalconWidgetDisObject& other);
            HalconWidgetDisObject& operator=(HalconWidgetDisObject&& other) noexcept;
        private:
            HalconCpp::HObject* _object;
        public:
            size_t id{0};
            std::string name{"Undefined"};
            bool isShow{true};
		};

        class HalconWidget : public QWidget
        {
            Q_OBJECT
        public:
            explicit HalconWidget(QWidget* parent = nullptr);
            ~HalconWidget() override;
        private:
            HalconCpp::HTuple* _halconWindowHandle{ nullptr };
        private:
			std::vector<HalconWidgetDisObject*> _halconObjects;
        private:
            void append_HObject(HalconWidgetDisObject* object);
        public:
			void appendHObject(const HalconWidgetDisObject& object);
			void clearHObject();
        private:
            void initialize_halconWindow();
            void close_halconWindow();
        private:
            void refresh_allObject();
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