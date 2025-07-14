#pragma once

#include"rqw_HalconUtilty.hpp"
#include"rqw_HalconWidgetDisObject.hpp"

#include <QWidget>
#include"opencv2/opencv.hpp"


namespace rw {
	namespace rqw {

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
            HalconWidgetDisObjectId appendHObject(const HalconWidgetObject& object);
            HalconWidgetDisObjectId appendHObject(HalconWidgetObject* object);
			void clearHObject();
        public:
            size_t width();
            size_t height();
        public:
            HalconWidgetObject* getObjectPtrById(HalconWidgetDisObjectId id);
            HalconWidgetObject getObjectById(HalconWidgetDisObjectId id);
            bool eraseObjectById(int id);
            bool eraseObjectsByType(HalconObjectType objectType);
        public:
            [[nodiscard]] std::vector<HalconWidgetDisObjectId> getAllIds() const;
            [[nodiscard]] std::vector<HalconWidgetDisObjectId> getIdsByType(HalconObjectType objectType) const;
            HalconWidgetDisObjectId getVailidAppendId();
        public:
			void updateWidget();
        public:
            HalconWidgetDisObjectId appendVerticalLine(int position, const PainterConfig& config = {});
            HalconWidgetDisObjectId appendHorizontalLine(int position, const PainterConfig& config = {});
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
            void display_HalconWidgetDisObject(HalconWidgetObject* object);
            void prepare_display(const PainterConfig & config);
        private:
            std::vector<HalconWidgetObject*> _halconObjects;
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
        private:
            std::vector<HalconCpp::HTuple> _shapeModelIds;
            void clear_shapeModels();
        public:
            HalconWidgetObject drawRect();
            HalconWidgetObject drawRect(PainterConfig config);
            HalconWidgetObject drawRect(PainterConfig config, bool isShow);
            HalconWidgetObject drawRect(PainterConfig config, double minHeight, double minWidth);
            HalconWidgetObject drawRect(PainterConfig config, bool isShow, double minHeight, double minWidth);
            HalconWidgetObject drawRect(PainterConfig config, bool isShow, double minHeight, double minWidth, bool& isDraw);
        public:
            HalconShapeId createShapeModel(HalconWidgetObject& rec);
            void shapeModel(const HalconShapeId& id);

        };

	}
}