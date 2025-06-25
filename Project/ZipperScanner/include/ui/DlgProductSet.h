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

private:
	std::vector<std::vector<int>> DOFindAllDuplicateIndices();
	void setDOErrorInfo(const std::vector<std::vector<int>>& index);
	void setDOErrorInfo(int index);
	std::vector<std::vector<int>> DIFindAllDuplicateIndices();
	void setDIErrorInfo(const std::vector<std::vector<int>>& index);
	void setDIErrorInfo(int index);

private:
	// IO监控页面的调试模式
	bool isDebugIO{ false };

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
	void cBox_takeCamera1Pictures_checked();
	void cBox_takeCamera2Pictures_checked();
	void cbox_qiyongerxiangji_checked();
	void cbox_qiyongyundongkongzhiqi_checked();
	void cbox_yundongkongzhiqichonglian_checked();

	// 基本参数
	void btn_shedingladaichangdu_clicked();
	void btn_xiangjichufachangdu_clicked();
	void btn_jishuqingling_clicked();
	void btn_changduqingling_clicked();
	void btn_shoudongladai_pressed();
	void btn_shoudongladai_released();
	void btn_shoudongchongkong_clicked();
	void btn_tuoji_clicked();
	void btn_xiangjichufa_clicked();
	void btn_shoudongsudu_clicked();
	void btn_meizhuanmaichongshu_clicked();
	void btn_zidongladaisudu_clicked();
	void btn_shedingzhouchang_clicked();
	void btn_yanshichongkong_clicked();
	void btn_chongkongshijian_clicked();
	void btn_yanshiziqi_clicked();
	void btn_jiajiansushijian_clicked();

	// 监控IO
	void cbox_DIqidonganniu_clicked(bool isChecked);
	void cbox_DIjiting_clicked(bool isChecked);
	void cbox_DIlalianlawan_clicked(bool isChecked);

	void cbox_DObujindianjimaichong_clicked(bool isChecked);
	void cbox_DOchongkong_clicked(bool isChecked);
	void cbox_DOtuoji_clicked(bool isChecked);
	void cbox_DOchufapaizhao_clicked(bool isChecked);

	void cbox_debugMode_checked(bool ischecked);


	// 设置IO
	void btn_setqidonganniu_clicked();
	void btn_setlalianlawan_clicked();
	void btn_setjiting_clicked();
	void btn_setbujindianjimaichong_clicked();
	void btn_setchongkong_clicked();
	void btn_settuoji_clicked();

	void tabWidget_indexChanged(int index);

	// 监控IO函数
	void monitorInPutSignal(size_t index, bool state);
	void monitorOutPutSignal(size_t index, bool state);

private:
	Ui::DlgProductSetClass *ui;
};

