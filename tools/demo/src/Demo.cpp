#include "Demo.h"

#include "ui_Demo.h"
#include<Algorithm>

Demo::Demo(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::DemoClass())
{
	ui->setupUi(this);
	halconWidget = new HalconWidget(this);
    ui->horizontalLayout->replaceWidget(ui->label, halconWidget);
	delete ui->label; 
}

Demo::~Demo()
{
	delete ui;
}

void Demo::ini()
{
    HalconCpp::HObject image;
    ReadImage(&image, "C:/Users/rw/Desktop/temp/4be85a13-4196-4ae1-bb3c-ccea8d1d27fa.png");

    halconWidget->setImage(image);

}

void Demo::resizeEvent(QResizeEvent* event)
{

	QMainWindow::resizeEvent(event);
}

