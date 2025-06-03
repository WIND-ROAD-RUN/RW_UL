#pragma once

#include <QDialog>
#include "ui_DlgProductSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductSetClass; };
QT_END_NAMESPACE

class DlgProductSet : public QDialog
{
	Q_OBJECT

public:
	DlgProductSet(QWidget *parent = nullptr);
	~DlgProductSet();

public:
	void build_ui();
	void read_config();
	void build_connect();

public:
	void saveDlgProductSetConfig();

private slots:
	void pbtn_close_clicked();

	void pbtn_tifeichixushijian1_clicked();
	void pbtn_yanchitifeishijian1_clicked();
	void pbtn_tifeichixushijian2_clicked();
	void pbtn_yanchitifeishijian2_clicked();
	void pbtn_shangxianwei1_clicked();
	void pbtn_xiaxianwei1_clicked();
	void pbtn_zuoxianwei1_clicked();
	void pbtn_youxianwei1_clicked();
	void pbtn_xiangsudangliang1_clicked();
	void pbtn_shangxianwei2_clicked();
	void pbtn_xiaxianwei2_clicked();
	void pbtn_zuoxianwei2_clicked();
	void pbtn_youxianwei2_clicked();
	void pbtn_xiangsudangliang2_clicked();
	void pbtn_qiangbaoguang_clicked();
	void pbtn_qiangzengyi_clicked();
	void pbtn_zhongbaoguang_clicked();
	void pbtn_ruobaoguang_clicked();
	void pbtn_zhongzengyi_clicked();
	void pbtn_ruozengyi_clicked();

	void cBox_takeNgPictures_checked();
	void cBox_takeMaskPictures_checked();
	void cBox_takeOkPictures_checked();
	void cbox_debugMode_checked();

private:
	Ui::DlgProductSetClass *ui;
};

