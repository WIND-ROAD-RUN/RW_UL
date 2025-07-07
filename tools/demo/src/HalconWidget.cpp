#include "HalconWidget.hpp"
#include <QResizeEvent>

using namespace HalconCpp;

HalconWidget::HalconWidget(QWidget* parent)
    : QWidget(parent)
{
    initializeHalconWindow();
}

HalconWidget::~HalconWidget()
{
    closeHalconWindow();
}

void HalconWidget::initializeHalconWindow()
{
    _halconWindow = new HalconCpp::HTuple();
    Hlong winId = this->winId();
    auto size = this->size();

    // 创建 Halcon 窗口
    //OpenWindow(0, 0, size.width(), size.height(), winId, "visible", "", &_halconWindow);
    OpenWindow(0, 0, size.width(), size.height(), winId, "visible", "", _halconWindow);
}

void HalconWidget::closeHalconWindow()
{
    // 关闭 Halcon 窗口
    if (_halconWindow != 0)
    {
        CloseWindow(*_halconWindow);
        delete _halconWindow;
        //CloseWindow(_halconWindow);
    }
}

void HalconWidget::resizeEvent(QResizeEvent* event)
{
    // 关闭并重新初始化 Halcon 窗口以适应新的大小
    closeHalconWindow();
    initializeHalconWindow();

    QWidget::resizeEvent(event);
}

void HalconWidget::displayImage(const HObject& image)
{
    // 获取图像的宽度和高度
    HTuple width, height;
    GetImageSize(image, &width, &height);

    // 获取窗口的宽度和高度
    int windowWidth = this->width();
    int windowHeight = this->height();

    // 计算图像显示的比例
    double scaleWidth = static_cast<double>(windowWidth) / width.D();
    double scaleHeight = static_cast<double>(windowHeight) / height.D();
    double scale;
    if (scaleWidth< scaleHeight)
    {
        scale = scaleWidth;
    }
    else
    {
        scale = scaleHeight;
    }

    // 计算显示区域
    int displayWidth = static_cast<int>(width.D() * scale);
    int displayHeight = static_cast<int>(height.D() * scale);
    int offsetX = (windowWidth - displayWidth) / 2;
    int offsetY = (windowHeight - displayHeight) / 2;

    // 设置显示区域
    SetPart(*_halconWindow, -offsetY, -offsetX, height.D() - offsetY - 1, width.D() - offsetX - 1);
    //SetPart(_halconWindow, -offsetY, -offsetX, height.D() - offsetY - 1, width.D() - offsetX - 1);

    // 显示图像
    DispObj(image, *_halconWindow);
    //DispObj(image, _halconWindow);

}