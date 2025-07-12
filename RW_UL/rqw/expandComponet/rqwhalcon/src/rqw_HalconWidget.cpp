#include"rqw_HalconWidget.hpp"
#include <QResizeEvent>
#include "halconcpp/HalconCpp.h"

#include"rqw_HalconUtilty.hpp"

namespace rw {
	namespace rqw
	{
        HalconWidget::HalconWidget(QWidget* parent)
            : QWidget(parent)
        {
            initialize_halconWindow();
        }

        HalconWidget::~HalconWidget()
        {
            clearHObject();
            close_halconWindow();
        }

        void HalconWidget::appendImage(const HalconCpp::HImage& image)
        {
            HalconCpp::HImage* newImage = new HalconCpp::HImage(image);
            append_HObject(newImage);
        }

        void HalconWidget::appendImage(const QImage& image)
        {
            HalconCpp::HImage hImage = QImageToHImage(image);
            auto newImage = new HalconCpp::HImage(hImage);
            append_HObject(newImage);
        }

        void HalconWidget::appendImage(const cv::Mat& mat)
        {
            HalconCpp::HImage hImage = CvMatToHImage(mat);
            auto newImage = new HalconCpp::HImage(hImage);
            append_HObject(newImage);
        }

        void HalconWidget::append_HObject(HalconCpp::HObject* object)
        {
            if (object == nullptr)
            {
                return;
            }
            _halconObjects.push_back(object);
            refresh_allObject();
        }

        void HalconWidget::appendHObject(const HalconCpp::HObject& object)
        {
			HalconCpp::HObject* newObject = new HalconCpp::HObject(object);
            _halconObjects.push_back(newObject);
            refresh_allObject();
        }

        void HalconWidget::clearHObject()
        {
            _halconObjects.clear();
            if (_halconWindowHandle)
            {
                ClearWindow(*_halconWindowHandle);
			}
        }

        void HalconWidget::refresh_allObject()
        {
            if (!_halconWindowHandle)
            {
                return;
            }

            // 清除窗口内容
            ClearWindow(*_halconWindowHandle);

            if (_halconObjects.empty())
            {
                return; // 如果没有对象需要显示，直接返回
            }

            // 获取窗口大小
            auto size = this->size();
            int windowWidth = size.width();
            int windowHeight = size.height();

            // 获取第一个对象的大小（假设所有对象的大小一致）
            HalconCpp::HTuple width, height;
            GetImageSize(*_halconObjects.front(), &width, &height);

            // 计算图像和窗口的宽高比
            double imgAspectRatio = static_cast<double>(width.I()) / height.I();
            double windowAspectRatio = static_cast<double>(windowWidth) / windowHeight;

            // 设置 Halcon 窗口铺满整个 QWidget
            SetWindowExtents(*_halconWindowHandle, 0, 0, windowWidth, windowHeight);

            // 计算显示区域
            int partWidth, partHeight, offsetX, offsetY;
            if (imgAspectRatio > windowAspectRatio)
            {
                // 图像更宽，以宽度为基准
                partWidth = width.I();
                partHeight = static_cast<int>(width.I() / windowAspectRatio);
                offsetX = 0;
                offsetY = (partHeight - height.I()) / 2;
            }
            else
            {
                // 图像更高，以高度为基准
                partHeight = height.I();
                partWidth = static_cast<int>(height.I() * windowAspectRatio);
                offsetX = (partWidth - width.I()) / 2;
                offsetY = 0;
            }

            // 设置显示区域，使图像居中等比例显示
            SetPart(*_halconWindowHandle, -offsetY, -offsetX, height.I() - 1 + offsetY, width.I() - 1 + offsetX);

            // 显示所有对象
            for (auto& object : _halconObjects)
            {
                DispObj(*object, *_halconWindowHandle);
            }
        }

        void HalconWidget::initialize_halconWindow()
        {
            _halconWindowHandle = new HalconCpp::HTuple();
            Hlong winId = this->winId();
            auto size = this->size();

            OpenWindow(0, 0, size.width(), size.height(), winId, "visible", "", _halconWindowHandle);
        }

        void HalconWidget::close_halconWindow()
        {
            if (_halconWindowHandle != nullptr)
            {
                CloseWindow(*_halconWindowHandle);
                delete _halconWindowHandle;
            }
        }

        void HalconWidget::wheelEvent(QWheelEvent* event)
        {
            if (rect().contains(event->position().toPoint())) { // 检查鼠标是否在 HalconWidget 内
                int delta = event->angleDelta().y(); // 获取滚轮滚动的角度
                double scaleFactor = (delta > 0) ? 1.1 : 0.9; // 缩放因子，向上滚动放大，向下滚动缩小

                // 获取鼠标在 HalconWidget 中的位置
                QPointF mousePos = event->position();
                int mouseX = static_cast<int>(mousePos.x());
                int mouseY = static_cast<int>(mousePos.y());

                // 获取当前显示区域
                HalconCpp::HTuple row1, col1, row2, col2;
                GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

                // 计算当前显示区域的宽高
                double currentWidth = col2.D() - col1.D() + 1;
                double currentHeight = row2.D() - row1.D() + 1;

                // 计算鼠标位置在图像中的相对位置
                double relativeX = col1.D() + (mouseX / static_cast<double>(width())) * currentWidth;
                double relativeY = row1.D() + (mouseY / static_cast<double>(height())) * currentHeight;

                // 计算新的显示区域
                double newWidth = currentWidth / scaleFactor;
                double newHeight = currentHeight / scaleFactor;
                double newCol1 = relativeX - (mouseX / static_cast<double>(width())) * newWidth;
                double newRow1 = relativeY - (mouseY / static_cast<double>(height())) * newHeight;
                double newCol2 = newCol1 + newWidth - 1;
                double newRow2 = newRow1 + newHeight - 1;

                // 清除窗口内容
                ClearWindow(*_halconWindowHandle);

                // 设置新的显示区域
                SetPart(*_halconWindowHandle, newRow1, newCol1, newRow2, newCol2);

                // 重新显示所有对象
                for (auto& object : _halconObjects)
                {
                    DispObj(*object, *_halconWindowHandle);
                }

                event->accept(); // 事件已处理
            }
            else {
                event->ignore(); // 事件未处理
            }
        }

        void HalconWidget::showEvent(QShowEvent* event)
        {
            refresh_allObject();
            QWidget::showEvent(event);
        }

        void HalconWidget::resizeEvent(QResizeEvent* event)
        {
            refresh_allObject();
            QWidget::resizeEvent(event);
        }

        void HalconWidget::mousePressEvent(QMouseEvent* event)
        {
            if (_isDrawingRect) {
                event->ignore(); // 如果正在绘制矩形，忽略鼠标事件
                return;
            }

            if (event->button() == Qt::LeftButton && rect().contains(event->pos())) {
                _isDragging = true;
                _lastMousePos = event->pos(); // 记录鼠标按下时的位置
                event->accept();
            }
            else {
                event->ignore();
            }
        }

        void HalconWidget::mouseMoveEvent(QMouseEvent* event)
        {
            if (_isDrawingRect) {
                event->ignore(); // 如果正在绘制矩形，忽略鼠标事件
                return;
            }

            if (_isDragging && _halconWindowHandle && !_halconObjects.empty()) {
                QPoint currentMousePos = event->pos();
                QPoint delta = currentMousePos - _lastMousePos; // 计算鼠标移动的偏移量

                // 获取当前显示区域
                HalconCpp::HTuple row1, col1, row2, col2;
                GetPart(*_halconWindowHandle, &row1, &col1, &row2, &col2);

                // 根据鼠标移动的偏移量调整显示区域
                double deltaX = -delta.x() * (col2.D() - col1.D() + 1) / width();
                double deltaY = -delta.y() * (row2.D() - row1.D() + 1) / height();

                double newCol1 = col1.D() + deltaX;
                double newCol2 = col2.D() + deltaX;
                double newRow1 = row1.D() + deltaY;
                double newRow2 = row2.D() + deltaY;

                // 设置新的显示区域
                SetPart(*_halconWindowHandle, newRow1, newCol1, newRow2, newCol2);

                // 清除窗口并重新显示所有对象
                ClearWindow(*_halconWindowHandle);
                for (auto& object : _halconObjects) {
                    DispObj(*object, *_halconWindowHandle);
                }

                _lastMousePos = currentMousePos; // 更新鼠标位置
                event->accept();
            }
            else {
                event->ignore();
            }
        }

        void HalconWidget::mouseReleaseEvent(QMouseEvent* event)
        {
            if (_isDrawingRect) {
                event->ignore(); // 如果正在绘制矩形，忽略鼠标事件
                return;
            }

            if (event->button() == Qt::LeftButton) {
                _isDragging = false; // 停止拖拽
                event->accept();
            }
            else {
                event->ignore();
            }
        }

        void HalconWidget::drawRect()
        {
            _isDrawingRect = true; // 开始绘制矩形
            HalconCpp::HTuple hv_Row1, hv_Column1, hv_Row2, hv_Column2;
            HalconCpp::HObject ho_Rectangle;

            // 调用 Halcon 的绘制矩形方法
            HalconCpp::DrawRectangle1(*_halconWindowHandle, &hv_Row1, &hv_Column1, &hv_Row2, &hv_Column2);

            // 生成矩形对象并显示
            HalconCpp::GenRectangle1(&ho_Rectangle, hv_Row1, hv_Column1, hv_Row2, hv_Column2);
            appendHObject(ho_Rectangle);
        	HalconCpp::DispObj(ho_Rectangle, *_halconWindowHandle);

            _isDrawingRect = false; // 绘制完成
        }
	}
}