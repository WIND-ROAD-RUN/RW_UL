#include "DlgProductScore.h"

#include "GlobalStruct.hpp"

DlgProductScore::DlgProductScore(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductScoreClass())
{
	ui->setupUi(this);

	build_ui();
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
	auto& globalConfig = GlobalStructDataZipper::getInstance().scoreConfig;
	// ³õÊ¼»¯²ÎÊý

	// È±ÑÀ
	ui->rbtn_queyaEnable->setChecked(globalConfig.queYa);
	ui->ptn_queyaSimilarity->setText(QString::number(globalConfig.queYaScore));
	ui->ptn_queyaArea->setText(QString::number(globalConfig.queYaArea));

	// ÌÌÉË
	ui->rbtn_tangshangEnable->setChecked(globalConfig.tangShang);
	ui->pbtn_tangshangSimilarity->setText(QString::number(globalConfig.tangShangScore));
	ui->pbtn_tangshangArea->setText(QString::number(globalConfig.tangShangArea));

	// ÔàÎÛ
	ui->rbtn_zangwuEnable->setChecked(globalConfig.zangWu);
	ui->pbtn_zangwuSimilarity->setText(QString::number(globalConfig.zangWuScore));
	ui->pbtn_zangwuArea->setText(QString::number(globalConfig.zangWuArea));
}

