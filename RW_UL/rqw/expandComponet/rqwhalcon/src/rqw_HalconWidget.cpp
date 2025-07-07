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
            initializeHalconWindow();
        }

        HalconWidget::~HalconWidget()
        {
            closeHalconWindow();
            if (_image)
            {
                delete _image;
            }
        }

        void HalconWidget::setImage(const HalconCpp::HImage& image)
        {
            HalconCpp::HImage* newImage = new HalconCpp::HImage(image);
            _image = newImage;
        }

        void HalconWidget::setImage(const QImage& image)
        {

            HalconCpp::HImage hImage = HalconImageConverter::QImageToHImage(image);
            if (_image)
            {
                delete _image;
            }
            _image = new HalconCpp::HImage(hImage);
			displayImg();
        }

        void HalconWidget::initializeHalconWindow()
        {
            _halconWindow = new HalconCpp::HTuple();
            Hlong winId = this->winId();
            auto size = this->size();

            OpenWindow(0, 0, size.width(), size.height(), winId, "visible", "", _halconWindow);
        }

        void HalconWidget::closeHalconWindow()
        {
            if (_halconWindow != nullptr)
            {
                CloseWindow(*_halconWindow);
                delete _halconWindow;
            }
        }

        void HalconWidget::displayImg()
        {
            if (!_image || !_halconWindow)
            {
                return;
            }

            // 获取窗口大小
            auto size = this->size();
            int windowWidth = size.width();
            int windowHeight = size.height();

            // 获取图像大小
            HalconCpp::HTuple width, height;
            GetImageSize(*_image, &width, &height);

            // 计算图像和窗口的宽高比
            double imgAspectRatio = static_cast<double>(width.I()) / height.I();
            double windowAspectRatio = static_cast<double>(windowWidth) / windowHeight;

            // 计算显示区域
            int displayWidth, displayHeight, offsetX, offsetY;
            if (imgAspectRatio > windowAspectRatio)
            {
                // 图像更宽，以宽度为基准
                displayWidth = windowWidth;
                displayHeight = static_cast<int>(windowWidth / imgAspectRatio);
                offsetX = 0;
                offsetY = (windowHeight - displayHeight) / 2;
            }
            else
            {
                // 图像更高，以高度为基准
                displayHeight = windowHeight;
                displayWidth = static_cast<int>(windowHeight * imgAspectRatio);
                offsetX = (windowWidth - displayWidth) / 2;
                offsetY = 0;
            }

            // 设置显示区域
            SetPart(*_halconWindow, 0, 0, height.I() - 1, width.I() - 1);

            // 调整窗口显示位置
            HalconCpp::SetWindowExtents(*_halconWindow, offsetX, offsetY, displayWidth, displayHeight);

            // 显示图像
            DispObj(*_image, *_halconWindow);
        }

        void HalconWidget::showEvent(QShowEvent* event)
        {
            displayImg();
            QWidget::showEvent(event);
        }

        void HalconWidget::resizeEvent(QResizeEvent* event)
        {
            if (!_image)
            {
                QWidget::resizeEvent(event);
                return;
            }

            HalconCpp::HTuple width, height;
            GetImageSize(*_image, &width, &height);

            double imgAspectRatio = static_cast<double>(width.I()) / height.I();

            const int minHeight = 100;
            const int minWidth = static_cast<int>(minHeight * imgAspectRatio);

            this->setMinimumSize(minWidth, minHeight);

            if (event->size().width() < minWidth || event->size().height() < minHeight)
            {
                return;
            }

            displayImg();
            QWidget::resizeEvent(event);
        }

	}
}