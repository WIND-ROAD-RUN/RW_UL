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
            explicit HalconWidgetDisObject(const QPixmap& pixmap);

            ~HalconWidgetDisObject();
        public:
            HalconWidgetDisObject(const HalconWidgetDisObject& other);
            HalconWidgetDisObject(HalconWidgetDisObject&& other) noexcept;
            HalconWidgetDisObject& operator=(const HalconWidgetDisObject& other);
            HalconWidgetDisObject& operator=(HalconWidgetDisObject&& other) noexcept;
        private:
            HalconCpp::HObject* _object;
        public:
            int id{0};
            std::string name{"Undefined"};
            bool isShow{true};
        public:
            bool has_value();
            HalconCpp::HObject* value();
        public:
            void release();
        public:
            void updateObject(const HalconCpp::HObject & object);
            void updateObject(HalconCpp::HObject* object);
		};

        class HalconWidget : public QWidget
        {
            Q_OBJECT
        public:
            explicit HalconWidget(QWidget* parent = nullptr);
            ~HalconWidget() override;
        private:
            HalconCpp::HTuple* _halconWindowHandle{ nullptr };
        public:
			void appendHObject(const HalconWidgetDisObject& object);
            void appendHObject(HalconWidgetDisObject * object);
			void clearHObject();
        public:
            HalconWidgetDisObject* getObjectPtrById(int id);
            HalconWidgetDisObject getObjectById(int id);
            bool eraseObjectById(int id);
        public:
			void updateWidget();
        protected:
            void wheelEvent(QWheelEvent* event) override; 
        protected:
            void showEvent(QShowEvent* event) override;
            void resizeEvent(QResizeEvent* event) override;
        protected:
            void mousePressEvent(QMouseEvent* event) override;
            void mouseMoveEvent(QMouseEvent* event) override;
            void mouseReleaseEvent(QMouseEvent* event) override;
        private:
            std::vector<HalconWidgetDisObject*> _halconObjects;
        private:
            void append_HObject(HalconWidgetDisObject* object);
        private:
            void initialize_halconWindow();
            void close_halconWindow();
        private:
            void refresh_allObject();
        private:
            bool _isDragging{ false }; 
            QPoint _lastMousePos;
            bool _isDrawingRect{ false };
        public slots:
            void drawRect();
        };

	}
}