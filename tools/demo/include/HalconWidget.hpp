#pragma once

#include <QWidget>
#include "halconcpp/HalconCpp.h"


class HalconWidget : public QWidget
{
    Q_OBJECT

public:
    explicit HalconWidget(QWidget* parent = nullptr);
    ~HalconWidget() override;

    void displayImage(const HalconCpp::HObject& image);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    HalconCpp::HTuple  *_halconWindow; 
    void initializeHalconWindow();
    void closeHalconWindow();
};