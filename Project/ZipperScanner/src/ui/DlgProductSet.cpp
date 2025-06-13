#include "DlgProductSet.h"

#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "NumberKeyboard.h"

DlgProductSet::DlgProductSet(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductSetClass())
{
	ui->setupUi(this);

	build_ui();

	build_connect();
}

DlgProductSet::~DlgProductSet()
{
	delete ui;
}

void DlgProductSet::build_ui()
{
	read_config();
}

void DlgProductSet::read_config()
{
	auto& globalConfig = GlobalStructDataZipper::getInstance().setConfig;

	// 剔废时间
	ui->pbtn_tifeichixushijian1->setText(QString::number(globalConfig.tiFeiChiXuShiJian1));
	ui->pbtn_yanchitifeishijian1->setText(QString::number(globalConfig.yanChiTiFeiShiJian1));
	ui->pbtn_tifeichixushijian2->setText(QString::number(globalConfig.tiFeiChiXuShiJian2));
	ui->pbtn_yanchitifeishijian2->setText(QString::number(globalConfig.yanChiTiFeiShiJian2));

	// 采图
	ui->cBox_takeCamera1Pictures->setChecked(globalConfig.takeWork1Pictures);
	ui->cBox_takeCamera2Pictures->setChecked(globalConfig.takeWork2Pictures);

	// 存图
	ui->cBox_takeNgPictures->setChecked(globalConfig.saveNGImg);
	ui->cBox_takeMaskPictures->setChecked(globalConfig.saveMaskImg);
	ui->cBox_takeOkPictures->setChecked(globalConfig.saveOKImg);

	// 一工位的限位与像素当量
	ui->pbtn_shangxianwei1->setText(QString::number(globalConfig.shangXianWei1));
	ui->pbtn_xiaxianwei1->setText(QString::number(globalConfig.xiaXianWei1));
	ui->pbtn_zuoxianwei1->setText(QString::number(globalConfig.zuoXianWei1));
	ui->pbtn_youxianwei1->setText(QString::number(globalConfig.youXianWei1));
	ui->pbtn_xiangsudangliang1->setText(QString::number(globalConfig.xiangSuDangLiang1));

	// 二工位的限位与像素当量
	ui->pbtn_shangxianwei2->setText(QString::number(globalConfig.shangXianWei2));
	ui->pbtn_xiaxianwei2->setText(QString::number(globalConfig.xiaXianWei2));
	ui->pbtn_zuoxianwei2->setText(QString::number(globalConfig.zuoXianWei2));
	ui->pbtn_youxianwei2->setText(QString::number(globalConfig.youXianWei2));
	ui->pbtn_xiangsudangliang2->setText(QString::number(globalConfig.xiangSuDangLiang2));

	// 光源
	ui->pbtn_qiangbaoguang->setText(QString::number(globalConfig.qiangBaoGuang));
	ui->pbtn_qiangzengyi->setText(QString::number(globalConfig.qiangZengYi));

	ui->pbtn_zhongbaoguang->setText(QString::number(globalConfig.zhongBaoGuang));
	ui->pbtn_zhongzengyi->setText(QString::number(globalConfig.zhongZengYi));

	ui->pbtn_ruobaoguang->setText(QString::number(globalConfig.ruoBaoGuang));
	ui->pbtn_ruozengyi->setText(QString::number(globalConfig.ruoZengYi));

	// 调试模式默认为关闭
	ui->cbox_debugMode->setChecked(globalConfig.debugMode);
}

void DlgProductSet::build_connect()
{
	QObject::connect(ui->pbtn_tifeichixushijian1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_tifeichixushijian1_clicked);
	QObject::connect(ui->pbtn_yanchitifeishijian1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_yanchitifeishijian1_clicked);
	QObject::connect(ui->pbtn_tifeichixushijian2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_tifeichixushijian2_clicked);
	QObject::connect(ui->pbtn_yanchitifeishijian2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_yanchitifeishijian2_clicked);
	QObject::connect(ui->pbtn_shangxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_shangxianwei1_clicked);
	QObject::connect(ui->pbtn_xiaxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiaxianwei1_clicked);
	QObject::connect(ui->pbtn_zuoxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zuoxianwei1_clicked);
	QObject::connect(ui->pbtn_youxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_youxianwei1_clicked);
	QObject::connect(ui->pbtn_xiangsudangliang1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiangsudangliang1_clicked);
	QObject::connect(ui->pbtn_shangxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_shangxianwei2_clicked);
	QObject::connect(ui->pbtn_xiaxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiaxianwei2_clicked);
	QObject::connect(ui->pbtn_zuoxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zuoxianwei2_clicked);
	QObject::connect(ui->pbtn_youxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_youxianwei2_clicked);
	QObject::connect(ui->pbtn_xiangsudangliang2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiangsudangliang2_clicked);
	QObject::connect(ui->pbtn_qiangbaoguang, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_qiangbaoguang_clicked);
	QObject::connect(ui->pbtn_qiangzengyi, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_qiangzengyi_clicked);
	QObject::connect(ui->pbtn_zhongbaoguang, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zhongbaoguang_clicked);
	QObject::connect(ui->pbtn_ruobaoguang, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_ruobaoguang_clicked);
	QObject::connect(ui->pbtn_zhongzengyi, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zhongzengyi_clicked);
	QObject::connect(ui->pbtn_ruozengyi, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_ruozengyi_clicked);
	QObject::connect(ui->cBox_takeNgPictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeNgPictures_checked);
	QObject::connect(ui->cBox_takeMaskPictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeMaskPictures_checked);
	QObject::connect(ui->cBox_takeOkPictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeOkPictures_checked);
	QObject::connect(ui->cbox_debugMode, &QCheckBox::clicked,
		this, &DlgProductSet::cbox_debugMode_checked);
	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_close_clicked);
	QObject::connect(ui->cBox_takeCamera1Pictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeCamera1Pictures_checked);
	QObject::connect(ui->cBox_takeCamera2Pictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeCamera2Pictures_checked);
}

void DlgProductSet::pbtn_close_clicked()
{
	auto& GlobalStructData = GlobalStructDataZipper::getInstance();
	GlobalStructData.saveDlgProductSetConfig();
	this->close();
}


void DlgProductSet::pbtn_tifeichixushijian1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_tifeichixushijian1->setText(value);
		globalStructSetConfig.tiFeiChiXuShiJian1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_yanchitifeishijian1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_yanchitifeishijian1->setText(value);
		globalStructSetConfig.yanChiTiFeiShiJian1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_tifeichixushijian2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_tifeichixushijian2->setText(value);
		globalStructSetConfig.tiFeiChiXuShiJian2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_yanchitifeishijian2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_yanchitifeishijian2->setText(value);
		globalStructSetConfig.yanChiTiFeiShiJian2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_shangxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_shangxianwei1->setText(value);
		globalStructSetConfig.shangXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiaxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiaxianwei1->setText(value);
		globalStructSetConfig.xiaXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_zuoxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_zuoxianwei1->setText(value);
		globalStructSetConfig.zuoXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_youxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_youxianwei1->setText(value);
		globalStructSetConfig.youXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiangsudangliang1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiangsudangliang1->setText(value);
		globalStructSetConfig.xiangSuDangLiang1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_shangxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_shangxianwei2->setText(value);
		globalStructSetConfig.shangXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiaxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiaxianwei2->setText(value);
		globalStructSetConfig.xiaXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_zuoxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_zuoxianwei2->setText(value);
		globalStructSetConfig.zuoXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_youxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_youxianwei2->setText(value);
		globalStructSetConfig.youXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiangsudangliang2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiangsudangliang2->setText(value);
		globalStructSetConfig.xiangSuDangLiang2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_qiangbaoguang_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 11 || value.toDouble() > 300)
		{
			QMessageBox::warning(this, "提示", "请输入11到300的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;

		ui->pbtn_qiangbaoguang->setText(value);
		globalStructSetConfig.qiangBaoGuang = value.toDouble();
		if (globalStructGeneralConfig.qiangGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setExposureTime(static_cast<size_t>(globalStructSetConfig.qiangBaoGuang));
			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setExposureTime(static_cast<size_t>(globalStructSetConfig.qiangBaoGuang));
			}
		}
	}
}

void DlgProductSet::pbtn_qiangzengyi_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 16)
		{
			QMessageBox::warning(this, "提示", "请输入0到16的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_qiangzengyi->setText(value);
		globalStructSetConfig.qiangZengYi = value.toDouble();
		if (globalStructGeneralConfig.qiangGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.qiangZengYi));
			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setGain(static_cast<size_t>(globalStructSetConfig.qiangZengYi));
			}
		}
	}
}

void DlgProductSet::pbtn_zhongbaoguang_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 11 || value.toDouble() > 300)
		{
			QMessageBox::warning(this, "提示", "请输入11到300的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_zhongbaoguang->setText(value);
		globalStructSetConfig.zhongBaoGuang = value.toDouble();
		if (globalStructGeneralConfig.zhongGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setExposureTime(static_cast<size_t>(globalStructSetConfig.zhongBaoGuang));
			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setExposureTime(static_cast<size_t>(globalStructSetConfig.zhongBaoGuang));
			}
		}
	}
}

void DlgProductSet::pbtn_ruobaoguang_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 11 || value.toDouble() > 300)
		{
			QMessageBox::warning(this, "提示", "请输入11到300的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_ruobaoguang->setText(value);
		globalStructSetConfig.ruoBaoGuang = value.toDouble();
		if (globalStructGeneralConfig.ruoGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setExposureTime(static_cast<size_t>(globalStructSetConfig.ruoBaoGuang));

			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setExposureTime(static_cast<size_t>(globalStructSetConfig.ruoBaoGuang));
			}
		}
	}
}

void DlgProductSet::pbtn_zhongzengyi_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 16)
		{
			QMessageBox::warning(this, "提示", "请输入0到16的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_zhongzengyi->setText(value);
		globalStructSetConfig.zhongZengYi = value.toDouble();
		if (globalStructGeneralConfig.zhongGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.zhongZengYi));

			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setGain(static_cast<size_t>(globalStructSetConfig.zhongZengYi));
			}
		}
	}
}

void DlgProductSet::pbtn_ruozengyi_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 16)
		{
			QMessageBox::warning(this, "提示", "请输入0到16的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_ruozengyi->setText(value);
		globalStructSetConfig.ruoZengYi = value.toDouble();
		if (globalStructGeneralConfig.ruoGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.ruoZengYi));

			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setGain(static_cast<size_t>(globalStructSetConfig.ruoZengYi));
			}
		}
	}
}

void DlgProductSet::cBox_takeNgPictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.saveNGImg = ui->cBox_takeNgPictures->isChecked();
}

void DlgProductSet::cBox_takeMaskPictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.saveMaskImg = ui->cBox_takeMaskPictures->isChecked();
}

void DlgProductSet::cBox_takeOkPictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.saveOKImg = ui->cBox_takeOkPictures->isChecked();
}

void DlgProductSet::cbox_debugMode_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.debugMode = ui->cbox_debugMode->isChecked();
}

void DlgProductSet::cBox_takeCamera1Pictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.takeWork1Pictures = ui->cBox_takeCamera1Pictures->isChecked();
}

void DlgProductSet::cBox_takeCamera2Pictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.takeWork2Pictures = ui->cBox_takeCamera2Pictures->isChecked();
}

