#pragma once

#include <QDialog>
#include "ui_DlgProductScore.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductScoreClass; };
QT_END_NAMESPACE

class DlgProductScoreSmartCroppingOfBags : public QDialog
{
	Q_OBJECT

public:
	DlgProductScoreSmartCroppingOfBags(QWidget *parent = nullptr);
	~DlgProductScoreSmartCroppingOfBags();

public:
	void build_ui();
	void read_config();
	void build_connect();

private slots:
    void btn_close_clicked();

    void ckb_heiba_checked();
    void btn_heibascore_clicked();
    void btn_heibaarea_clicked();

    void ckb_shudang_checked();
    void btn_shudangscore_clicked();
    void btn_shudangarea_clicked();

    void ckb_huapo_checked();
    void btn_huaposcore_clicked();
    void btn_huapoarea_clicked();

    void ckb_jietou_checked();
    void btn_jietouscore_clicked();
    void btn_jietouarea_clicked();

    void ckb_guasi_checked();
    void btn_guasiscore_clicked();
    void btn_guasiarea_clicked();

    void ckb_podong_checked();
    void btn_podongscore_clicked();
    void btn_podongarea_clicked();

    void ckb_zangwu_checked();
    void btn_zangwuscore_clicked();
    void btn_zangwuarea_clicked();

    void ckb_noshudang_checked();
    void btn_noshudangscore_clicked();
    void btn_noshudangarea_clicked();

    void ckb_modian_checked();
    void btn_modianscore_clicked();
    void btn_modianarea_clicked();

    void ckb_loumo_checked();
    void btn_loumoscore_clicked();
    void btn_loumoarea_clicked();

    void ckb_xishudang_checked();
    void btn_xishudangscore_clicked();
    void btn_xishudangarea_clicked();

    void ckb_erweima_checked();
    void btn_erweimascore_clicked();
    void btn_erweimaarea_clicked();

    void ckb_damodian_checked();
    void btn_damodianscore_clicked();
    void btn_damodianarea_clicked();

    void ckb_kongdong_checked();
    void btn_kongdongscore_clicked();
    void btn_kongdongarea_clicked();

    void ckb_sebiao_checked();
    void btn_sebiaoscore_clicked();
    void btn_sebiaoarea_clicked();

    void ckb_yinshuaquexian_checked();
    void btn_yinshuaquexianscore_clicked();
    void btn_yinshuaquexianarea_clicked();

	void ckb_xiaopodong_checked();
	void btn_xiaopodongscore_clicked();
	void btn_xiaopodongarea_clicked();

	void ckb_jiaodai_checked();
	void btn_jiaodaiscore_clicked();
	void btn_jiaodaiarea_clicked();

private:
	Ui::DlgProductScoreClass *ui;
};

