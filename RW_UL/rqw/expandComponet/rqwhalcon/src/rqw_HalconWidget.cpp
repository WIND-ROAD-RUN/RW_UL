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

        void HalconWidget::setImage(const cv::Mat& mat)
        {
            HalconCpp::HImage hImage = HalconImageConverter::CVMatToHImage(mat);
            if (_image)
            {
                delete _image;
            }
            _image = new HalconCpp::HImage(hImage);
            displayImg();
        }

        void HalconWidget::initializeHalconWindow()
        {
            _halconWindowHandle = new HalconCpp::HTuple();
            Hlong winId = this->winId();
            auto size = this->size();

            OpenWindow(0, 0, size.width(), size.height(), winId, "visible", "", _halconWindowHandle);
        }

        void HalconWidget::closeHalconWindow()
        {
            if (_halconWindowHandle != nullptr)
            {
                CloseWindow(*_halconWindowHandle);
                delete _halconWindowHandle;
            }
        }

        void HalconWidget::displayImg()
        {
            if (!_image || !_halconWindowHandle)
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

            // 显示图像
            DispObj(*_image, *_halconWindowHandle);
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

        void HalconWidget::drawRect()
        {
            HalconCpp::HTuple  hv_WindowHandle, hv_Row1, hv_Column1;
            HalconCpp::HTuple  hv_Row2, hv_Column2, hv_ModelID, hv_Row, hv_Column, hv_Angle, hv_Score, hv_HomMat2D;
            HalconCpp::HObject ho_Rectangle, ho_ImageReduced, ho_ModelContours, ho_ContoursAffineTrans;
            HalconCpp::DrawRectangle1(*_halconWindowHandle, &hv_Row1, &hv_Column1, &hv_Row2, &hv_Column2);
          
            HalconCpp::SetDraw(*_halconWindowHandle, "margin");
           
            HalconCpp::SetLineWidth(*_halconWindowHandle, 5);

            HalconCpp::GenRectangle1(&ho_Rectangle, hv_Row1, hv_Column1, hv_Row2, hv_Column2);

            HalconCpp::DispObj(ho_Rectangle, *_halconWindowHandle);
        }
	}
}