#include "Demo.h"

#include "ui_Demo.h"
#include<Algorithm>
#include"opencv2/opencv.hpp"

Demo::Demo(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::DemoClass())
{
	ui->setupUi(this);
	build_ui();
	build_connect();
}

Demo::~Demo()
{
	delete ui;
}

void Demo::build_ui()
{
	halconWidget = new rw::rqw::HalconWidget(this);
	ui->verticalLayout->replaceWidget(ui->label, halconWidget);
	delete ui->label;
}

void Demo::build_connect()
{
	QObject::connect(ui->pushButton, &QPushButton::clicked,
		this, &Demo::pushButton_clicked);
}

void Demo::ini()
{
  
	/*HalconCpp::ReadImage(&image, "C:/Users/rw/Desktop/temp/4be85a13-4196-4ae1-bb3c-ccea8d1d27fa.png");
	Rgb1ToGray(image, &image);*/

	//QImage image("C:/Users/rw/Desktop/temp/4be85a13-4196-4ae1-bb3c-ccea8d1d27fa.png");

	auto image = cv::imread("C:/Users/rw/Desktop/temp/4be85a13-4196-4ae1-bb3c-ccea8d1d27fa.png");
	rw::rqw::HalconWidgetDisObject object(image);
	object.isShow = true;
	object.id = 0;
	auto a=object.has_value();
    halconWidget->appendHObject(object);
	auto id=halconWidget->getMinValidAppendId();
	halconWidget->drawHorizontalLine(10);
}

void Demo::resizeEvent(QResizeEvent* event)
{

	QMainWindow::resizeEvent(event);
}

void Demo::pushButton_clicked()
{
	halconWidget->drawRect();

	////画一个矩形
	//HalconCpp::HTuple  hv_WindowHandle, hv_Row1, hv_Column1;
	//HalconCpp::HTuple  hv_Row2, hv_Column2, hv_ModelID, hv_Row, hv_Column, hv_Angle, hv_Score, hv_HomMat2D;
	//HalconCpp::HObject ho_Rectangle, ho_ImageReduced, ho_ModelContours, ho_ContoursAffineTrans;
	//DrawRectangle1(*halconWidget->_halconWindowHandle, &hv_Row1, &hv_Column1, &hv_Row2, &hv_Column2);
	////画矩形
	//GenRectangle1(&ho_Rectangle, hv_Row1, hv_Column1, hv_Row2, hv_Column2);


	//// 显示图像
	//DispObj(ho_Rectangle, *halconWidget->_halconWindowHandle);

	////图像类型变量qimage    数值类型变量

	//ReduceDomain(image, ho_Rectangle, &ho_ImageReduced);

	////创建模板
	//CreateShapeModel(ho_ImageReduced, "auto", -0.39, 0.79, "auto", "auto", "use_polarity",
	//	"auto", "auto", &hv_ModelID);
	////获取模板轮廓
	//GetShapeModelContours(&ho_ModelContours, hv_ModelID, 1);

	//FindShapeModel(ho_ImageReduced, hv_ModelID, -0.39, 0.79, 0.5, 1, 0.5, "least_squares",
	//	0, 0.9, &hv_Row, &hv_Column, &hv_Angle, &hv_Score);
	////数组不为空说明找到模板
	//if ((hv_Row.TupleLength()) > 0)
	//{

	//	//位置转换函数
	//	VectorAngleToRigid(0, 0, 0, hv_Row, hv_Column, hv_Angle, &hv_HomMat2D);
	//	HalconCpp::SetColor(*halconWidget->_halconWindowHandle, "blue");
	//	//set_color(WindowHandle, 'blue')
	//	//移动
	//	AffineTransContourXld(ho_ModelContours, &ho_ContoursAffineTrans, hv_HomMat2D);


	//}

	//
	//
	//// 显示图像
	//DispObj(image, *halconWidget->_halconWindowHandle);

	//// 显示图像
	//DispObj(ho_ContoursAffineTrans, *halconWidget->_halconWindowHandle);


}

