#pragma once

#include <QDialog>
#include "ui_DlgProductSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductSetClass; };
QT_END_NAMESPACE

class DlgProductSetSmartCroppingOfBags : public QDialog
{
	Q_OBJECT

public:
	DlgProductSetSmartCroppingOfBags(QWidget *parent = nullptr);
	~DlgProductSetSmartCroppingOfBags();

public:
	void build_ui();
	void read_config();
	void build_connect();

private slots:
	void pbtn_close_clicked();

	void btn_zidongpingbifanwei_clicked();
	void btn_jiange_clicked();
	void btn_pingjunmaichong1_clicked();
	void btn_maichongxinhao1_clicked();
	void btn_hanggao1_clicked();
	void btn_daichang1_clicked();
	void btn_daichangxishu1_clicked();
	void btn_guasijuli1_clicked();
	void btn_zuixiaodaichang1_clicked();
	void btn_zuidadaichang1_clicked();
	void btn_baisedailiangdufanweiMin1_clicked();
	void btn_baisedailiangdufanweiMax1_clicked();
	void btn_daokoudaoxiangjijuli1_clicked();
	void btn_tifeiyanshi1_clicked();
	void btn_baojingyanshi1_clicked();
	void btn_tifeishijian1_clicked();
	void btn_chuiqiyanshi1_clicked();
	void btn_dudaiyanshi1_clicked();
	void btn_chuiqishijian1_clicked();
	void btn_dudaishijian1_clicked();
	void btn_maichongxishu1_clicked();
	void btn_xiangjizengyi1_clicked();
	void btn_houfenpinqi1_clicked();
	void btn_chengfaqi1_clicked();
	void btn_qiedaoxianshangpingbi1_clicked();
	void btn_qiedaoxianxiapingbi1_clicked();
	void btn_yansedailiangdufanweiMin1_clicked();
	void btn_yansedailiangdufanweiMax1_clicked();

	void ckb_xiaopodong_checked();
	void ckb_tiqiantifei_checked();
	void ckb_xiangjitiaoshi_checked();
	void ckb_qiyonger_checked();
	void ckb_xiangjizengyi_checked();

private:
	Ui::DlgProductSetClass *ui;
};

