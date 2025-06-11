#include "DlgProductScore.h"

#include "GlobalStruct.hpp"

DlgProductScoreSmartCroppingOfBags::DlgProductScoreSmartCroppingOfBags(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductScoreClass())
{
	ui->setupUi(this);

	build_ui();

	build_connect();
}

DlgProductScoreSmartCroppingOfBags::~DlgProductScoreSmartCroppingOfBags()
{
	delete ui;
}

void DlgProductScoreSmartCroppingOfBags::build_ui()
{
	read_config();
}

void DlgProductScoreSmartCroppingOfBags::read_config()
{
	auto& ScoreConfig = GlobalStructDataSmartCroppingOfBags::getInstance().scoreConfig;

	ui->ckb_heiba->setChecked(ScoreConfig.heiba);
	ui->btn_heibascore->setText(QString::number(ScoreConfig.heibascore));
	ui->btn_heibaarea->setText(QString::number(ScoreConfig.heibaarea));
}

void DlgProductScoreSmartCroppingOfBags::build_connect()
{

}

