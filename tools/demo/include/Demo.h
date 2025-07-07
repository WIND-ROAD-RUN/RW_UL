#pragma once

#include <QMainWindow>
#include"halconcpp/HalconCpp.h"
#include"HalconWidget.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class DemoClass; };
QT_END_NAMESPACE

class Demo : public QMainWindow
{
	Q_OBJECT
public:
	HalconWidget *halconWidget;
public:
	Demo(QWidget *parent = nullptr);
	~Demo() override;

public:
	void ini();

protected:
	void resizeEvent(QResizeEvent* event) override;

private:
	Ui::DemoClass *ui;
};

