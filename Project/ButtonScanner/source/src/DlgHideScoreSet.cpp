#include "stdafx.h"
#include "DlgHideScoreSet.h"

#include "GlobalStruct.h"
#include"NumberKeyboard.h"

DlgHideScoreSet::DlgHideScoreSet(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgHideScoreSetClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
}

DlgHideScoreSet::~DlgHideScoreSet()
{
	delete ui;
}

void DlgHideScoreSet::build_ui()
{
	readConfig();
}

void DlgHideScoreSet::build_connect()
{
	QObject::connect(ui->pbtn_close, &QPushButton::clicked, this,
		&DlgHideScoreSet::pbtn_close_clicked);
	QObject::connect(ui->pbtn_outsideDiameterScore, &QPushButton::clicked, this,
		&DlgHideScoreSet::pbtn_outsideDiameterScore_clicked);
	QObject::connect(ui->pbtn_forAndAgainstScore, &QPushButton::clicked, this,
		&DlgHideScoreSet::pbtn_forAndAgainstScore_clicked);
}

void DlgHideScoreSet::readConfig()
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	ui->pbtn_outsideDiameterScore->setText(QString::number(GlobalStructData.dlgHideScoreSetConfig.outsideDiameterScore));
	ui->pbtn_forAndAgainstScore->setText(QString::number(GlobalStructData.dlgHideScoreSetConfig.forAndAgainstScore));
}

void DlgHideScoreSet::pbtn_close_clicked()
{
	this->hide();
}

void DlgHideScoreSet::pbtn_outsideDiameterScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_outsideDiameterScore->setText(value);
		GlobalStructData.dlgHideScoreSetConfig.outsideDiameterScore = value.toDouble();
	}
}

void DlgHideScoreSet::pbtn_forAndAgainstScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_forAndAgainstScore->setText(value);
		GlobalStructData.dlgHideScoreSetConfig.forAndAgainstScore = value.toDouble();
	}
}