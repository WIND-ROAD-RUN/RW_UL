#include "DlgProductSet.h"

#include "GlobalStruct.hpp"

DlgProductSetSmartCroppingOfBags::DlgProductSetSmartCroppingOfBags(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductSetClass())
{
	ui->setupUi(this);

	build_ui();

	build_connect();
}

DlgProductSetSmartCroppingOfBags::~DlgProductSetSmartCroppingOfBags()
{
	delete ui;
}

void DlgProductSetSmartCroppingOfBags::build_ui()
{
	read_config();
}

void DlgProductSetSmartCroppingOfBags::read_config()
{
	auto& globalConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;

	// 公共参数
	ui->btn_zidongpingbifanwei->setText(QString::number(globalConfig.zidongpingbifanwei));
	ui->ckb_xiaopodong->setChecked(globalConfig.xiaopodong);
	ui->ckb_tiqiantifei->setChecked(globalConfig.tiqiantifei);
	ui->ckb_xiangjitiaoshi->setChecked(globalConfig.xiangjitiaoshi);
	ui->ckb_qiyonger->setChecked(globalConfig.qiyonger);
	ui->btn_jiange->setText(QString::number(globalConfig.jiange));

	// 1相机参数
	ui->btn_pingjunmaichong1->setText(QString::number(globalConfig.pingjunmaichong1));
	ui->btn_maichongxinhao1->setText(QString::number(globalConfig.maichongxinhao1));
	ui->btn_hanggao1->setText(QString::number(globalConfig.hanggao1));
	ui->btn_daichang1->setText(QString::number(globalConfig.daichang1));
	ui->btn_daichangxishu1->setText(QString::number(globalConfig.daichangxishu1));
	ui->btn_guasijuli1->setText(QString::number(globalConfig.guasijuli1));
	ui->btn_zuixiaodaichang1->setText(QString::number(globalConfig.zuixiaodaichang1));
	ui->btn_zuidadaichang1->setText(QString::number(globalConfig.zuidadaichang1));
	ui->btn_baisedailiangdufanweiMin1->setText(QString::number(globalConfig.baisedailiangdufanweimin1));
	ui->btn_baisedailiangdufanweiMax1->setText(QString::number(globalConfig.baisedailiangdufanweimax1));

	ui->btn_daokoudaoxiangjijuli1->setText(QString::number(globalConfig.daokoudaoxiangjiluli1));
	ui->btn_tifeiyanshi1->setText(QString::number(globalConfig.tifeiyanshi1));
	ui->btn_tifeishijian1->setText(QString::number(globalConfig.tifeishijian1));
	ui->btn_baojingyanshi1->setText(QString::number(globalConfig.baojingyanshi1));
	ui->btn_baojingshijian1->setText(QString::number(globalConfig.baojingshijian1));
	ui->btn_chuiqiyanshi1->setText(QString::number(globalConfig.chuiqiyanshi1));
	ui->btn_chuiqishijian1->setText(QString::number(globalConfig.chuiqishijian1));
	ui->btn_dudaiyanshi1->setText(QString::number(globalConfig.dudaiyanshi1));
	ui->btn_dudaishijian1->setText(QString::number(globalConfig.dudaishijian1));
	ui->btn_maichongxishu1->setText(QString::number(globalConfig.maichongxishu1));
	ui->ckb_xiangjizengyi->setChecked(globalConfig.isxiangjizengyi1);
	ui->btn_xiangjizengyi1->setText(QString::number(globalConfig.xiangjizengyi1));
	ui->btn_houfenpinqi1->setText(QString::number(globalConfig.houfenpinqi1));
	ui->btn_chengfaqi1->setText(QString::number(globalConfig.chengfaqi1));
	ui->btn_qiedaoxianshangpingbi1->setText(QString::number(globalConfig.qiedaoxianshangpingbi1));
	ui->btn_qiedaoxianxiapingbi1->setText(QString::number(globalConfig.qiedaoxianxiapingbi1));
	ui->btn_yansedailiangdufanweiMin1->setText(QString::number(globalConfig.yansedailiangdufanweimin1));
	ui->btn_yansedailiangdufanweiMax1->setText(QString::number(globalConfig.yansedailiangdufanweimax1));
}

void DlgProductSetSmartCroppingOfBags::build_connect()
{

}

