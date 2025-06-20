#pragma once

#include <QDialog>
#include "ui_DlgProductSet.h"
#include "Utilty.hpp"


namespace rw
{
	namespace rqw
	{
		class MonitorZMotionIOStateThread;
	}

}

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductSetClass; };
QT_END_NAMESPACE

class DlgProductSetSmartCroppingOfBags : public QDialog
{
	Q_OBJECT

public:
	DlgProductSetSmartCroppingOfBags(QWidget *parent = nullptr);
	~DlgProductSetSmartCroppingOfBags();
private:
	std::unique_ptr<rw::rqw::MonitorZMotionIOStateThread> _monitorZmotion;
private:
	std::vector<std::vector<int>> DOFindAllDuplicateIndices();
	void setDOErrorInfo(const std::vector<std::vector<int>>& index);
	void setDOErrorInfo(int index);
private:
	// IO监控页面的调试模式
	bool isDebugIO{ false };
protected:
	void showEvent(QShowEvent*) override;
public:
	void build_ui();
	void read_config();
	void build_connect();
public slots:
	void onUpdateMonitorRunningStateInfo(MonitorRunningStateInfo info);
private slots:
	void pbtn_close_clicked();

	void btn_zidongpingbifanwei_clicked();
	void btn_jiange_clicked();
	void btn_daichangxishu1_clicked();
	void btn_guasijuli1_clicked();
	void btn_zuixiaodaichang1_clicked();
	void btn_zuidadaichang1_clicked();
	void btn_baisedailiangdufanweiMin1_clicked();
	void btn_baisedailiangdufanweiMax1_clicked();
	void btn_daokoudaoxiangjijuli1_clicked();
	void btn_xiangjibaoguang1_clicked();
	void btn_tifeiyanshi1_clicked();
	void btn_baojingyanshi1_clicked();
	void btn_baojingshijian1_clicked();
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
	void btn_daichang1_clicked();

	void ckb_xiaopodong_checked();
	void ckb_tiqiantifei_checked();
	void ckb_xiangjitiaoshi_checked();
	void ckb_qiyonger_checked();
	void ckb_yundongkongzhiqichonglian_checked();
	void ckb_xiangjizengyi_checked();

	void btn_qiedao_clicked();
	void btn_chuiqi_clicked();
	void btn_baojinghongdeng_clicked();
	void btn_yadai_clicked();
	void btn_tifei_clicked();

	void ckb_debugIO_checked(bool ischecked);
	void ckb_qiedao_checked(bool ischecked);
	void ckb_chuiqi_checked(bool ischecked);
	void ckb_baojinghongdeng_checked(bool ischecked);
	void ckb_yadai_checked(bool ischecked);
	void ckb_tifei_checked(bool ischecked);

	void tabWidget_indexChanged(int index);

private:
	Ui::DlgProductSetClass *ui;
public slots:
	void onDIState(size_t index, bool state);
	void onDOState(size_t index, bool state);
};


