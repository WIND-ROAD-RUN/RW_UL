#include "DlgProductScore.h"

#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "NumberKeyboard.h"

DlgProductScore::DlgProductScore(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductScoreClass())
{
	ui->setupUi(this);

	build_ui();

	build_connect();
}

DlgProductScore::~DlgProductScore()
{
	delete ui;
}

void DlgProductScore::build_ui()
{
	read_config();
}

void DlgProductScore::read_config()
{
	auto& globalScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	// 初始化参数

	// 缺牙
	ui->rbtn_queyaEnable->setChecked(globalScoreConfig.queYa);
	ui->ptn_queyaSimilarity->setText(QString::number(globalScoreConfig.queYaScore));
	ui->ptn_queyaArea->setText(QString::number(globalScoreConfig.queYaArea));

	// 烫伤
	ui->rbtn_tangshangEnable->setChecked(globalScoreConfig.tangShang);
	ui->pbtn_tangshangSimilarity->setText(QString::number(globalScoreConfig.tangShangScore));
	ui->pbtn_tangshangArea->setText(QString::number(globalScoreConfig.tangShangArea));

	// 脏污
	ui->rbtn_zangwuEnable->setChecked(globalScoreConfig.zangWu);
	ui->pbtn_zangwuSimilarity->setText(QString::number(globalScoreConfig.zangWuScore));
	ui->pbtn_zangwuArea->setText(QString::number(globalScoreConfig.zangWuArea));
}

void DlgProductScore::build_connect()
{
	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgProductScore::pbtn_close_clicked);
	QObject::connect(ui->rbtn_queyaEnable, &QRadioButton::clicked,
		this, &DlgProductScore::rbtn_queyaEnable_checked);
	QObject::connect(ui->ptn_queyaSimilarity, &QPushButton::clicked,
		this, &DlgProductScore::ptn_queyaSimilarity_clicked);
	QObject::connect(ui->ptn_queyaArea, &QPushButton::clicked,
		this, &DlgProductScore::ptn_queyaArea_clicked);
	QObject::connect(ui->rbtn_tangshangEnable, &QRadioButton::clicked,
		this, &DlgProductScore::rbtn_tangshangEnable_checked);
	QObject::connect(ui->pbtn_tangshangSimilarity, &QPushButton::clicked,
		this, &DlgProductScore::ptn_tangshangSimilarity_clicked);
	QObject::connect(ui->pbtn_tangshangArea, &QPushButton::clicked,
		this, &DlgProductScore::ptn_tangshangArea_clicked);
	QObject::connect(ui->rbtn_zangwuEnable, &QRadioButton::clicked,
		this, &DlgProductScore::rbtn_zangwuEnable_checked);
	QObject::connect(ui->pbtn_zangwuSimilarity, &QPushButton::clicked,
		this, &DlgProductScore::ptn_zangwuSimilarity_clicked);
	QObject::connect(ui->pbtn_zangwuArea, &QPushButton::clicked,
		this, &DlgProductScore::ptn_zangwuArea_clicked);
}

void DlgProductScore::pbtn_close_clicked()
{
	auto& GlobalStructData = GlobalStructDataZipper::getInstance();
	GlobalStructData.saveDlgProductScoreConfig();
	this->close();
}

void DlgProductScore::rbtn_queyaEnable_checked()
{
	auto& globalScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	globalScoreConfig.queYa = ui->rbtn_queyaEnable->isChecked();
	GlobalStructDataZipper::getInstance().saveDlgProductScoreConfig();
}

void DlgProductScore::ptn_queyaSimilarity_clicked()
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
		auto& globalStructScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
		ui->ptn_queyaSimilarity->setText(value);
		globalStructScoreConfig.queYaScore = value.toDouble();
	}
}

void DlgProductScore::ptn_queyaArea_clicked()
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
		auto& globalStructScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
		ui->ptn_queyaArea->setText(value);
		globalStructScoreConfig.queYaArea = value.toDouble();
	}
}

void DlgProductScore::rbtn_tangshangEnable_checked()
{
	auto& globalScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	globalScoreConfig.tangShang = ui->rbtn_tangshangEnable->isChecked();
	GlobalStructDataZipper::getInstance().saveDlgProductScoreConfig();
}

void DlgProductScore::ptn_tangshangSimilarity_clicked()
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
		auto& globalStructScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
		ui->pbtn_tangshangSimilarity->setText(value);
		globalStructScoreConfig.tangShangScore = value.toDouble();
	}
}

void DlgProductScore::ptn_tangshangArea_clicked()
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
		auto& globalStructScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
		ui->pbtn_tangshangArea->setText(value);
		globalStructScoreConfig.tangShangArea = value.toDouble();
	}
}

void DlgProductScore::rbtn_zangwuEnable_checked()
{
	auto& globalScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	globalScoreConfig.zangWu = ui->rbtn_zangwuEnable->isChecked();
	GlobalStructDataZipper::getInstance().saveDlgProductScoreConfig();
}

void DlgProductScore::ptn_zangwuSimilarity_clicked()
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
		auto& globalStructScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
		ui->pbtn_zangwuSimilarity->setText(value);
		globalStructScoreConfig.zangWuScore = value.toDouble();
	}
}

void DlgProductScore::ptn_zangwuArea_clicked()
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
		auto& globalStructScoreConfig = GlobalStructDataZipper::getInstance().scoreConfig;
		ui->pbtn_zangwuArea->setText(value);
		globalStructScoreConfig.zangWuArea = value.toDouble();
	}
}


