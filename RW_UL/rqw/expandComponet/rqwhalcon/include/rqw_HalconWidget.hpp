#pragma once

#include <QWidget>
#include"opencv2/opencv.hpp"
#include"rqw_rqwColor.hpp"

namespace HalconCpp
{
	class HTuple;
	class HObject;
    class HImage;
}


namespace rw {
	namespace rqw {
        using HalconWidgetDisObjectId = int;

        class HalconWidget;

        struct PainterConfig
        {
        public:
            PainterConfig() = default;

            PainterConfig(const RQWColor& color, int thickness)
                : color(color), thickness(thickness) {
            }

            PainterConfig(const PainterConfig& other) = default;

            PainterConfig(PainterConfig&& other) noexcept = default;

            PainterConfig& operator=(const PainterConfig& other) = default;

            PainterConfig& operator=(PainterConfig&& other) noexcept = default;

            ~PainterConfig() = default;
        public:
            RQWColor color{ RQWColor::Black };
			int thickness{ 3 };
        };

        struct HalconWidgetDisObject
        {
            friend HalconWidget;
        public:
            enum class ObjectType {
                Image,
                Region,
                Undefined
			};
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
			// Object properties :default id=0 for image, id>0 is other, < 0 is inside
            HalconWidgetDisObjectId id{0};
            std::string descrption{"Undefined"};
            bool isShow{true};
			PainterConfig painterConfig;
            ObjectType type;
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
            HalconCpp::HTuple* Handle();
        public:
			void appendHObject(const HalconWidgetDisObject& object);
            void appendHObject(HalconWidgetDisObject * object);
			void clearHObject();
        public:
            size_t width();
            size_t height();
        public:
            HalconWidgetDisObject* getObjectPtrById(HalconWidgetDisObjectId id);
            HalconWidgetDisObject getObjectById(HalconWidgetDisObjectId id);
            bool eraseObjectById(int id);
            bool eraseObjectsByType(HalconWidgetDisObject::ObjectType objectType);
        public:
            [[nodiscard]] std::vector<HalconWidgetDisObjectId> getAllIds() const;
            [[nodiscard]] std::vector<HalconWidgetDisObjectId> getIdsByType(HalconWidgetDisObject::ObjectType objectType) const;
            HalconWidgetDisObjectId getVailidAppendId();
        public:
			void updateWidget();
        public:
            void appendVerticalLine(int position, const PainterConfig& config={});
            void appendHorizontalLine(int position, const PainterConfig& config={});
        public:
            bool setObjectVisible(HalconWidgetDisObjectId id,const bool visible);
        public:
            bool isDrawing();
        protected:
            void showEvent(QShowEvent* event) override;
            void resizeEvent(QResizeEvent* event) override;
        protected:
            void wheelEvent(QWheelEvent* event) override;
        protected:
            void mousePressEvent(QMouseEvent* event) override;
            void mouseMoveEvent(QMouseEvent* event) override;
            void mouseReleaseEvent(QMouseEvent* event) override;
        private:
            void display_HalconWidgetDisObject(HalconWidgetDisObject* object);
            void prepare_display(const PainterConfig & config);
        private:
            std::vector<HalconWidgetDisObject*> _halconObjects;
        private:
            void initialize_halconWindow();
            void close_halconWindow();
        private:
            void refresh_allObject();
        private:
            bool _isDragging{ false }; 
            QPoint _lastMousePos;
        private:
            bool _isDrawingRect{ false };
        /*public slots:
            void drawRect();*/
        public:
            HalconWidgetDisObject drawRect();
            HalconWidgetDisObject drawRect(PainterConfig config);
            HalconWidgetDisObject drawRect(PainterConfig config, bool isShow);
            HalconWidgetDisObject drawRect(PainterConfig config, double minHeight, double minWidth);
            HalconWidgetDisObject drawRect(PainterConfig config, bool isShow, double minHeight, double minWidth);
            HalconWidgetDisObject drawRect(PainterConfig config, bool isShow, double minHeight, double minWidth, bool& isDraw);

        public:
            void shapeModel(HalconWidgetDisObject & rec);
            void study();
        };

	}
}