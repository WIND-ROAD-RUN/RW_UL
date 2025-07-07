#pragma once

#include <QWidget>

namespace HalconCpp
{
    class HTuple;
	class HObject;
}

class HalconWidget : public QWidget
{
    Q_OBJECT

public:
    explicit HalconWidget(QWidget* parent = nullptr);
    ~HalconWidget() override;

    void setImage(const HalconCpp::HObject& image);
private:
    HalconCpp::HTuple  *_halconWindow{ nullptr };
    HalconCpp::HObject* _image{nullptr};
private:
    void initializeHalconWindow();
    void closeHalconWindow();
    void displayImg();

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
};