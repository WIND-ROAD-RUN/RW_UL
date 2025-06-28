#include "DlgIOTrigger.h"

#include <QMessageBox>
#include <QtConcurrent/qtconcurrentrun.h>

#include "GlobalStruct.hpp"

DlgIOTrigger::DlgIOTrigger(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgIOTriggerClass())
{
	ui->setupUi(this);

	build_connect();
}

DlgIOTrigger::~DlgIOTrigger()
{
	delete ui;
}

void DlgIOTrigger::build_connect()
{
	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgIOTrigger::pbtn_close_clicked);
	QObject::connect(ui->btn_shoudongladai, &QPushButton::pressed,
		this, &DlgIOTrigger::btn_shoudongladai_pressed);
	QObject::connect(ui->btn_shoudongladai, &QPushButton::released,
		this, &DlgIOTrigger::btn_shoudongladai_released);
	QObject::connect(ui->btn_tuoji, &QPushButton::clicked,
		this, &DlgIOTrigger::btn_tuoji_clicked);
	QObject::connect(ui->btn_xiangjichufa, &QPushButton::clicked,
		this, &DlgIOTrigger::btn_xiangjichufa_clicked);
	QObject::connect(ui->btn_shoudongchongkong, &QPushButton::clicked,
		this, &DlgIOTrigger::btn_shoudongchongkong_clicked);
}

void DlgIOTrigger::pbtn_close_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	bool isChongKongSet = globalStruct.zmotion.setIOOut(ControlLines::chongkongOUT, false);

	// 冲孔
	if (!isChongKongSet)
	{
		QMessageBox::warning(this, "警告", "关闭冲孔失败!");
	}

	// 脱机
	bool isTuoJiSet = globalStruct.zmotion.setIOOut(ControlLines::tuojiOut, false);
	if (!isTuoJiSet)
	{
		QMessageBox::warning(this, "警告", "关闭脱机失败!");
	}

	this->close();
}

void DlgIOTrigger::btn_shoudongladai_pressed()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto setConfig = globalStruct.setConfig;
	auto value = setConfig.meizhuanmaichongshu / setConfig.shedingzhouchang;

	// 启动电机
	auto isAxisType = globalStruct.zmotion.setAxisType(0, 1);
	double unit = value;
	auto isAxisPulse = globalStruct.zmotion.setAxisPulse(0, unit);
	double acc = setConfig.jiajiansushijian;
	auto isAxisAcc = globalStruct.zmotion.setAxisAcc(0, acc);
	auto isAxisDec = globalStruct.zmotion.setAxisDec(0, acc);
	double speed = setConfig.shoudongsudu;
	auto isAxisRunSpeed = globalStruct.zmotion.setAxisRunSpeed(0, speed);
	auto isAxisRun = globalStruct.zmotion.setAxisRun(0, -1);

	if (!isAxisType || !isAxisPulse || !isAxisAcc || !isAxisDec || !isAxisRunSpeed || !isAxisRun)
	{
		QMessageBox::warning(this, "警告", "电机参数设置失败");
	}
}

void DlgIOTrigger::btn_shoudongladai_released()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	// 停止电机
	bool isStop = globalStruct.zmotion.stopAllAxis();

	if (!isStop)
	{
		QMessageBox::warning(this, "警告", "手动拉带取消失败!");
	}
}

void DlgIOTrigger::btn_shoudongchongkong_clicked()
{
	auto future = QtConcurrent::run([this]() {
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto chongkongshijian = globalStruct.setConfig.chongkongshijian;
		auto yanchichongkongshijian = globalStruct.setConfig.yanshichongkong;
		QThread::msleep(yanchichongkongshijian);
		bool isSet = globalStruct.zmotion.SetIOOut(2, ControlLines::chongkongOUT, true, chongkongshijian);

		if (!isSet) {
			QMetaObject::invokeMethod(this, [this]() {
				QMessageBox::warning(this, "警告", "手动冲孔失败!");
				}, Qt::QueuedConnection);
		}
		});
}

void DlgIOTrigger::btn_tuoji_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	bool isGetTuoJiOut = false;
	bool tuojiState = globalStruct.zmotion.getIOOut(ControlLines::tuojiOut, isGetTuoJiOut);
	bool isSuccess = globalStruct.zmotion.setIOOut(ControlLines::tuojiOut, !tuojiState);
	if (!isSuccess)
	{
		QMessageBox::warning(this, "警告", "设置脱机失败!");
	}

	if (!tuojiState)
	{
		ui->btn_tuoji->setText("已脱机...");
		ui->btn_tuoji->setStyleSheet(
			"QPushButton {"
			"border-color: #388E3C;"
			"background-color: #4CAF50;"
			"color: #fff;"
			"font-weight: bold;"
			"}"
		);
	}
	else
	{
		ui->btn_tuoji->setText("脱机");
		ui->btn_tuoji->setStyleSheet(
			"QPushButton {"
			"padding: 6px 14px;"
			"border: 2px solid #CCC;"
			"border-radius: 4px;"
			"background-color: red;"
			"color: #444;"
			"}"
		);
	}
}

void DlgIOTrigger::btn_xiangjichufa_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	bool isXiangJiChuFaSet = globalStruct.zmotion.SetIOOut(3, 10, true, 100);

	if (!isXiangJiChuFaSet)
	{
		QMessageBox::warning(this, "警告", "相机触发失败!");
	}
}

