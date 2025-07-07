#include "HalconWidget.hpp"
#include <QResizeEvent>
#include "halconcpp/HalconCpp.h"

using namespace HalconCpp;

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

void HalconWidget::setImage(const HalconCpp::HObject& image)
{
	HalconCpp::HObject* newImage = new HalconCpp::HObject(image);
    _image = newImage;
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
    HTuple width, height;
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
    // 检查窗口大小是否真的发生变化
    if (event->size() != event->oldSize())
    {
        // 仅在窗口大小变化时更新显示
        displayImg();
    }
   
    /*closeHalconWindow();
    initializeHalconWindow();*/
    displayImg();

    QWidget::resizeEvent(event);
}
