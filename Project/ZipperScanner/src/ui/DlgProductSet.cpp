#include "DlgProductSet.h"
#include "GlobalStruct.hpp"

DlgProductSet::DlgProductSet(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductSetClass())
{
	ui->setupUi(this);

	read_config();
}

DlgProductSet::~DlgProductSet()
{
	delete ui;
}

void DlgProductSet::build_ui()
{
	
}

void DlgProductSet::read_config()
{
	auto& globalConfig = GlobalStructDataZipper::getInstance().setConfig;

	// 剔废时间
	ui->pbtn_tifeichixushijian1->setText(QString::number(globalConfig.tiFeiChiXuShiJian1));
	ui->pbtn_yanchitifeishijian1->setText(QString::number(globalConfig.yanChiTiFeiShiJian1));
	ui->pbtn_tifeichixushijian2->setText(QString::number(globalConfig.tiFeiChiXuShiJian2));
	ui->pbtn_yanchitifeishijian2->setText(QString::number(globalConfig.yanChiTiFeiShiJian2));

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

	// 调试模式
	ui->cbox_debugMode->setChecked(globalConfig.debugMode);
}

