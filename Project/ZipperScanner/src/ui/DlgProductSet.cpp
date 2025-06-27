#include "DlgProductSet.h"

#include <QMessageBox>

#include "GlobalStruct.hpp"
#include "NumberKeyboard.h"

DlgProductSet::DlgProductSet(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductSetClass())
{
	ui->setupUi(this);

	build_ui();

	build_connect();
}

DlgProductSet::~DlgProductSet()
{
	delete ui;
}

void DlgProductSet::build_ui()
{
	read_config();

	// 查看是否有相同的输入输出IO
	auto indicesDO = DOFindAllDuplicateIndices();
	setDOErrorInfo(indicesDO);
	auto indicesDI = DIFindAllDuplicateIndices();
	setDIErrorInfo(indicesDI);
}

void DlgProductSet::read_config()
{
	auto& globalConfig = GlobalStructDataZipper::getInstance().setConfig;

	// 剔废时间
	ui->pbtn_tifeichixushijian1->setText(QString::number(globalConfig.tiFeiChiXuShiJian1));
	ui->pbtn_tifeijuli1->setText(QString::number(globalConfig.tifeijuli1));
	ui->pbtn_tifeichixushijian2->setText(QString::number(globalConfig.tiFeiChiXuShiJian2));
	ui->pbtn_tifeijuli2->setText(QString::number(globalConfig.tifeijuli2));

	// 采图
	ui->cBox_takeCamera1Pictures->setChecked(globalConfig.takeWork1Pictures);
	ui->cBox_takeCamera2Pictures->setChecked(globalConfig.takeWork2Pictures);

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

	// 调试模式默认为关闭
	ui->cbox_debugMode->setChecked(globalConfig.debugMode);

	// 基本功能
	ui->cbox_qiyongerxiangji->setChecked(globalConfig.qiyongerxiangji);
	ui->cbox_qiyongyundongkongzhiqi->setChecked(globalConfig.qiyongyundongkongzhiqi);
	ui->cbox_yundongkongzhiqichonglian->setChecked(globalConfig.yundongkongzhiqichonglian);

	// 基本参数
	ui->btn_shedingladaichangdu->setText(QString::number(globalConfig.shedingladaichangdu));
	ui->btn_chongkongjishu->setText(QString::number(globalConfig.chongkongjishu));
	ui->btn_dangqianchangdu->setText(QString::number(globalConfig.dangqianchangdu));
	ui->btn_xiangjichufachangdu->setText(QString::number(globalConfig.xiangjichufachangdu));
	ui->btn_yanshichongkong->setText(QString::number(globalConfig.yanshichongkong));
	ui->btn_chongkongshijian->setText(QString::number(globalConfig.chongkongshijian));
	ui->btn_yanshiziqi->setText(QString::number(globalConfig.yanshiziqi));
	ui->btn_jiajiansushijian->setText(QString::number(globalConfig.jiajiansushijian));
	ui->btn_shoudongsudu->setText(QString::number(globalConfig.shoudongsudu));
	ui->btn_meizhuanmaichongshu->setText(QString::number(globalConfig.meizhuanmaichongshu));
	ui->btn_zidongladaisudu->setText(QString::number(globalConfig.zidongladaisudu));
	ui->btn_shedingzhouchang->setText(QString::number(globalConfig.shedingzhouchang));

	// 设置IO
	ui->btn_setqidonganniu->setText(QString::number(globalConfig.qidonganniuIN));
	ui->btn_setlalianlawan->setText(QString::number(globalConfig.lalianlawanIN));
	ui->btn_setjiting->setText(QString::number(globalConfig.jitingIN));
	ui->btn_setbujindianjimaichong->setText(QString::number(globalConfig.bujindianjimaichongOUT));
	ui->btn_setchongkong->setText(QString::number(globalConfig.chongkongOUT));
	ui->btn_settuoji->setText(QString::number(globalConfig.tuojiOUT));

	// 默认显示第一个
	ui->tabWidget->setCurrentIndex(0);
}

void DlgProductSet::build_connect()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	QObject::connect(ui->pbtn_tifeichixushijian1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_tifeichixushijian1_clicked);
	QObject::connect(ui->pbtn_tifeijuli1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_yanchitifeishijian1_clicked);
	QObject::connect(ui->pbtn_tifeichixushijian2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_tifeichixushijian2_clicked);
	QObject::connect(ui->pbtn_tifeijuli2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_yanchitifeishijian2_clicked);
	QObject::connect(ui->pbtn_shangxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_shangxianwei1_clicked);
	QObject::connect(ui->pbtn_xiaxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiaxianwei1_clicked);
	QObject::connect(ui->pbtn_zuoxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zuoxianwei1_clicked);
	QObject::connect(ui->pbtn_youxianwei1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_youxianwei1_clicked);
	QObject::connect(ui->pbtn_xiangsudangliang1, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiangsudangliang1_clicked);
	QObject::connect(ui->pbtn_shangxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_shangxianwei2_clicked);
	QObject::connect(ui->pbtn_xiaxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiaxianwei2_clicked);
	QObject::connect(ui->pbtn_zuoxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zuoxianwei2_clicked);
	QObject::connect(ui->pbtn_youxianwei2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_youxianwei2_clicked);
	QObject::connect(ui->pbtn_xiangsudangliang2, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_xiangsudangliang2_clicked);
	QObject::connect(ui->pbtn_qiangbaoguang, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_qiangbaoguang_clicked);
	QObject::connect(ui->pbtn_qiangzengyi, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_qiangzengyi_clicked);
	QObject::connect(ui->pbtn_zhongbaoguang, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zhongbaoguang_clicked);
	QObject::connect(ui->pbtn_ruobaoguang, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_ruobaoguang_clicked);
	QObject::connect(ui->pbtn_zhongzengyi, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_zhongzengyi_clicked);
	QObject::connect(ui->pbtn_ruozengyi, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_ruozengyi_clicked);
	QObject::connect(ui->cBox_takeNgPictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeNgPictures_checked);
	QObject::connect(ui->cBox_takeMaskPictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeMaskPictures_checked);
	QObject::connect(ui->cBox_takeOkPictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeOkPictures_checked);
	QObject::connect(ui->cbox_debugMode, &QCheckBox::clicked,
		this, &DlgProductSet::cbox_debugMode_checked);
	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_close_clicked);
	QObject::connect(ui->cBox_takeCamera1Pictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeCamera1Pictures_checked);
	QObject::connect(ui->cBox_takeCamera2Pictures, &QCheckBox::clicked,
		this, &DlgProductSet::cBox_takeCamera2Pictures_checked);
	QObject::connect(ui->cbox_qiyongerxiangji, &QCheckBox::clicked,
		this, &DlgProductSet::cbox_qiyongerxiangji_checked);
	QObject::connect(ui->cbox_qiyongyundongkongzhiqi, &QCheckBox::clicked,
		this, &DlgProductSet::cbox_qiyongyundongkongzhiqi_checked);
	QObject::connect(ui->cbox_yundongkongzhiqichonglian, &QCheckBox::clicked,
		this, &DlgProductSet::cbox_yundongkongzhiqichonglian_checked);

	// 基本参数
	QObject::connect(ui->btn_shedingladaichangdu, &QPushButton::clicked,
		this, &DlgProductSet::btn_shedingladaichangdu_clicked);
	QObject::connect(ui->btn_xiangjichufachangdu, &QPushButton::clicked,
		this, &DlgProductSet::btn_xiangjichufachangdu_clicked);
	QObject::connect(ui->btn_jishuqingling, &QPushButton::clicked,
		this, &DlgProductSet::btn_jishuqingling_clicked);
	QObject::connect(ui->btn_changduqingling, &QPushButton::clicked,
		this, &DlgProductSet::btn_changduqingling_clicked);
	QObject::connect(ui->btn_shoudongladai, &QPushButton::pressed,
		this, &DlgProductSet::btn_shoudongladai_pressed);
	QObject::connect(ui->btn_shoudongladai, &QPushButton::released,
		this, &DlgProductSet::btn_shoudongladai_released);
	QObject::connect(ui->btn_shoudongchongkong, &QPushButton::clicked,
		this, &DlgProductSet::btn_shoudongchongkong_clicked);
	QObject::connect(ui->btn_tuoji, &QPushButton::clicked,
		this, &DlgProductSet::btn_tuoji_clicked);
	QObject::connect(ui->btn_xiangjichufa, &QPushButton::clicked,
		this, &DlgProductSet::btn_xiangjichufa_clicked);
	QObject::connect(ui->btn_shoudongsudu, &QPushButton::clicked,
		this, &DlgProductSet::btn_shoudongsudu_clicked);
	QObject::connect(ui->btn_meizhuanmaichongshu, &QPushButton::clicked,
		this, &DlgProductSet::btn_meizhuanmaichongshu_clicked);
	QObject::connect(ui->btn_zidongladaisudu, &QPushButton::clicked,
		this, &DlgProductSet::btn_zidongladaisudu_clicked);
	QObject::connect(ui->btn_shedingzhouchang, &QPushButton::clicked,
		this, &DlgProductSet::btn_shedingzhouchang_clicked);
	QObject::connect(ui->btn_yanshichongkong, &QPushButton::clicked,
		this, &DlgProductSet::btn_yanshichongkong_clicked);
	QObject::connect(ui->btn_chongkongshijian, &QPushButton::clicked,
		this, &DlgProductSet::btn_chongkongshijian_clicked);
	QObject::connect(ui->btn_yanshiziqi, &QPushButton::clicked,
		this, &DlgProductSet::btn_yanshiziqi_clicked);
	QObject::connect(ui->btn_jiajiansushijian, &QPushButton::clicked,
		this, &DlgProductSet::btn_jiajiansushijian_clicked);

	// 监控IO
	QObject::connect(ui->cbox_DIqidonganniu, &QPushButton::clicked,
		this, &DlgProductSet::cbox_DIqidonganniu_clicked);
	QObject::connect(ui->cbox_DIjiting, &QPushButton::clicked,
		this, &DlgProductSet::cbox_DIjiting_clicked);
	QObject::connect(ui->cbox_DIlalianlawan, &QPushButton::clicked,
		this, &DlgProductSet::cbox_DIlalianlawan_clicked);
	QObject::connect(ui->cbox_DObujindianjimaichong, &QPushButton::clicked,
		this, &DlgProductSet::cbox_DObujindianjimaichong_clicked);
	QObject::connect(ui->cbox_DOchongkong, &QPushButton::clicked,
		this, &DlgProductSet::cbox_DOchongkong_clicked);
	QObject::connect(ui->cbox_DOtuoji, &QPushButton::clicked,
		this, &DlgProductSet::cbox_DOtuoji_clicked);
	QObject::connect(ui->cbox_DOchufapaizhao, &QPushButton::clicked,
		this, &DlgProductSet::cbox_DOchufapaizhao_clicked);
	QObject::connect(ui->tabWidget, &QTabWidget::currentChanged,
		this, &DlgProductSet::tabWidget_indexChanged);

	// 设置IO
	QObject::connect(ui->btn_setqidonganniu, &QPushButton::clicked,
		this, &DlgProductSet::btn_setqidonganniu_clicked);
	QObject::connect(ui->btn_setlalianlawan, &QPushButton::clicked,
		this, &DlgProductSet::btn_setlalianlawan_clicked);
	QObject::connect(ui->btn_setjiting, &QPushButton::clicked,
		this, &DlgProductSet::btn_setjiting_clicked);
	QObject::connect(ui->btn_setbujindianjimaichong, &QPushButton::clicked,
		this, &DlgProductSet::btn_setbujindianjimaichong_clicked);
	QObject::connect(ui->btn_setchongkong, &QPushButton::clicked,
		this, &DlgProductSet::btn_setchongkong_clicked);
	QObject::connect(ui->btn_settuoji, &QPushButton::clicked,
		this, &DlgProductSet::btn_settuoji_clicked);


	// 连接监控IO信号
	QObject::connect(&globalStruct, &GlobalStructDataZipper::emit_InPutSignal,
		this, &DlgProductSet::monitorInPutSignal);
	QObject::connect(&globalStruct, &GlobalStructDataZipper::emit_OutPutSignal,
		this, &DlgProductSet::monitorOutPutSignal);
}

std::vector<std::vector<int>> DlgProductSet::DOFindAllDuplicateIndices()
{
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;
	std::vector<int> values = {
		setConfig.bujindianjimaichongOUT,
		setConfig.chongkongOUT,
		setConfig.tuojiOUT
	};

	std::unordered_map<int, std::vector<int>> valueToIndices;
	for (size_t i = 0; i < values.size(); ++i) {
		valueToIndices[values[i]].push_back(static_cast<int>(i));
	}

	std::vector<std::vector<int>> result;
	std::set<int> used; // 防止重复分组
	for (const auto& pair : valueToIndices) {
		if (pair.second.size() > 1) {
			// 只收集未被收录过的index组
			bool alreadyUsed = false;
			for (int idx : pair.second) {
				if (used.count(idx)) {
					alreadyUsed = true;
					break;
				}
			}
			if (!alreadyUsed) {
				result.push_back(pair.second);
				used.insert(pair.second.begin(), pair.second.end());
			}
		}
	}
	return result;
}

void DlgProductSet::setDIErrorInfo(const std::vector<std::vector<int>>& index)
{
	ui->lb_qidonganniu->clear();
	ui->lb_lalianlawan->clear();
	ui->lb_jiting->clear();

	for (const auto& classic : index)
	{
		for (const auto& item : classic)
		{
			setDIErrorInfo(item);
		}
	}
}

void DlgProductSet::setDIErrorInfo(int index)
{
	QString text = "重复数值";
	switch (index)
	{
	case 0:
		ui->lb_qidonganniu->setText(text);
		break;
	case 1:
		ui->lb_lalianlawan->setText(text);
		break;
	case 2:
		ui->lb_jiting->setText(text);
		break;
	}
}

void DlgProductSet::closeAllIOBtn()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	bool isChongKongSet = globalStruct.zmotion.setIOOut(ControlLines::chongkongOUT, false);

	// 冲孔
	if (!isChongKongSet)
	{
		QMessageBox::warning(this, "警告", "手动冲孔失败!");
	}

	// 脱机
	bool isTuoJiSet = globalStruct.zmotion.setIOOut(ControlLines::tuojiOut, false);
	if (!isTuoJiSet)
	{
		QMessageBox::warning(this, "警告", "设置脱机失败!");
	}
}

void DlgProductSet::setDOErrorInfo(const std::vector<std::vector<int>>& index)
{
	ui->lb_bujindianjimaichong->clear();
	ui->lb_chongkong->clear();
	ui->lb_tuoji->clear();

	for (const auto& classic : index)
	{
		for (const auto& item : classic)
		{
			setDOErrorInfo(item);
		}
	}
}

void DlgProductSet::setDOErrorInfo(int index)
{
	QString text = "重复数值";
	switch (index)
	{
	case 0:
		ui->lb_bujindianjimaichong->setText(text);
		break;
	case 1:
		ui->lb_chongkong->setText(text);
		break;
	case 2:
		ui->lb_tuoji->setText(text);
		break;
	}
}

std::vector<std::vector<int>> DlgProductSet::DIFindAllDuplicateIndices()
{
	auto& setConfig = GlobalStructDataZipper::getInstance().setConfig;
	std::vector<int> values = {
		setConfig.qidonganniuIN,
		setConfig.lalianlawanIN,
		setConfig.jitingIN
	};

	std::unordered_map<int, std::vector<int>> valueToIndices;
	for (size_t i = 0; i < values.size(); ++i) {
		valueToIndices[values[i]].push_back(static_cast<int>(i));
	}

	std::vector<std::vector<int>> result;
	std::set<int> used; // 防止重复分组
	for (const auto& pair : valueToIndices) {
		if (pair.second.size() > 1) {
			// 只收集未被收录过的index组
			bool alreadyUsed = false;
			for (int idx : pair.second) {
				if (used.count(idx)) {
					alreadyUsed = true;
					break;
				}
			}
			if (!alreadyUsed) {
				result.push_back(pair.second);
				used.insert(pair.second.begin(), pair.second.end());
			}
		}
	}
	return result;
}

void DlgProductSet::pbtn_close_clicked()
{
	auto& GlobalStructData = GlobalStructDataZipper::getInstance();

	// 关闭所有可以点动的IO按钮
	closeAllIOBtn();

	// 关闭监控IO线程
	GlobalStructDataZipper::getInstance()._isUpdateMonitorInfo = false;
	GlobalStructDataZipper::getInstance().monitorZMotionMonitorThread.setRunning(false);

	GlobalStructData.saveDlgProductSetConfig();
	this->close();
}


void DlgProductSet::pbtn_tifeichixushijian1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_tifeichixushijian1->setText(value);
		globalStructSetConfig.tiFeiChiXuShiJian1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_yanchitifeishijian1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_tifeijuli1->setText(value);
		globalStructSetConfig.tifeijuli1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_tifeichixushijian2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_tifeichixushijian2->setText(value);
		globalStructSetConfig.tiFeiChiXuShiJian2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_yanchitifeishijian2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_tifeijuli2->setText(value);
		globalStructSetConfig.tifeijuli2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_shangxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_shangxianwei1->setText(value);
		globalStructSetConfig.shangXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiaxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiaxianwei1->setText(value);
		globalStructSetConfig.xiaXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_zuoxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_zuoxianwei1->setText(value);
		globalStructSetConfig.zuoXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_youxianwei1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_youxianwei1->setText(value);
		globalStructSetConfig.youXianWei1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiangsudangliang1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiangsudangliang1->setText(value);
		globalStructSetConfig.xiangSuDangLiang1 = value.toDouble();
	}
}

void DlgProductSet::pbtn_shangxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_shangxianwei2->setText(value);
		globalStructSetConfig.shangXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiaxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiaxianwei2->setText(value);
		globalStructSetConfig.xiaXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_zuoxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_zuoxianwei2->setText(value);
		globalStructSetConfig.zuoXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_youxianwei2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_youxianwei2->setText(value);
		globalStructSetConfig.youXianWei2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_xiangsudangliang2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
		ui->pbtn_xiangsudangliang2->setText(value);
		globalStructSetConfig.xiangSuDangLiang2 = value.toDouble();
	}
}

void DlgProductSet::pbtn_qiangbaoguang_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 11 || value.toDouble() > 300)
		{
			QMessageBox::warning(this, "提示", "请输入11到300的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;

		ui->pbtn_qiangbaoguang->setText(value);
		globalStructSetConfig.qiangBaoGuang = value.toDouble();
		if (globalStructGeneralConfig.qiangGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setExposureTime(static_cast<size_t>(globalStructSetConfig.qiangBaoGuang));
			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setExposureTime(static_cast<size_t>(globalStructSetConfig.qiangBaoGuang));
			}
		}
	}
}

void DlgProductSet::pbtn_qiangzengyi_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 16)
		{
			QMessageBox::warning(this, "提示", "请输入0到16的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_qiangzengyi->setText(value);
		globalStructSetConfig.qiangZengYi = value.toDouble();
		if (globalStructGeneralConfig.qiangGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.qiangZengYi));
			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setGain(static_cast<size_t>(globalStructSetConfig.qiangZengYi));
			}
		}
	}
}

void DlgProductSet::pbtn_zhongbaoguang_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 11 || value.toDouble() > 300)
		{
			QMessageBox::warning(this, "提示", "请输入11到300的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_zhongbaoguang->setText(value);
		globalStructSetConfig.zhongBaoGuang = value.toDouble();
		if (globalStructGeneralConfig.zhongGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setExposureTime(static_cast<size_t>(globalStructSetConfig.zhongBaoGuang));
			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setExposureTime(static_cast<size_t>(globalStructSetConfig.zhongBaoGuang));
			}
		}
	}
}

void DlgProductSet::pbtn_ruobaoguang_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 11 || value.toDouble() > 300)
		{
			QMessageBox::warning(this, "提示", "请输入11到300的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_ruobaoguang->setText(value);
		globalStructSetConfig.ruoBaoGuang = value.toDouble();
		if (globalStructGeneralConfig.ruoGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setExposureTime(static_cast<size_t>(globalStructSetConfig.ruoBaoGuang));

			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setExposureTime(static_cast<size_t>(globalStructSetConfig.ruoBaoGuang));
			}
		}
	}
}

void DlgProductSet::pbtn_zhongzengyi_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 16)
		{
			QMessageBox::warning(this, "提示", "请输入0到16的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_zhongzengyi->setText(value);
		globalStructSetConfig.zhongZengYi = value.toDouble();
		if (globalStructGeneralConfig.zhongGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.zhongZengYi));

			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setGain(static_cast<size_t>(globalStructSetConfig.zhongZengYi));
			}
		}
	}
}

void DlgProductSet::pbtn_ruozengyi_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 16)
		{
			QMessageBox::warning(this, "提示", "请输入0到16的数值");
			return;
		}
		auto& globalStruct = GlobalStructDataZipper::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& globalStructGeneralConfig = globalStruct.generalConfig;
		ui->pbtn_ruozengyi->setText(value);
		globalStructSetConfig.ruoZengYi = value.toDouble();
		if (globalStructGeneralConfig.ruoGuang == true)
		{
			if (globalStruct.camera1)
			{
				globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.ruoZengYi));

			}
			if (globalStruct.camera2)
			{
				globalStruct.camera2->setGain(static_cast<size_t>(globalStructSetConfig.ruoZengYi));
			}
		}
	}
}

void DlgProductSet::cBox_takeNgPictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.saveNGImg = ui->cBox_takeNgPictures->isChecked();
}

void DlgProductSet::cBox_takeMaskPictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.saveMaskImg = ui->cBox_takeMaskPictures->isChecked();
}

void DlgProductSet::cBox_takeOkPictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.saveOKImg = ui->cBox_takeOkPictures->isChecked();
}

void DlgProductSet::cbox_debugMode_checked(bool ischecked)
{
	auto& isDebug = GlobalStructDataZipper::getInstance().setConfig.debugMode;
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	isDebug = ischecked;
	isDebugIO = ischecked;
	if (isDebugIO)
	{
		ui->cbox_DIqidonganniu->setChecked(false);
		ui->cbox_DIjiting->setChecked(false);
		ui->cbox_DIlalianlawan->setChecked(false);
		ui->cbox_DObujindianjimaichong->setChecked(false);
		ui->cbox_DOchongkong->setChecked(false);
		ui->cbox_DOtuoji->setChecked(false);
		ui->cbox_DOchufapaizhao->setChecked(false);

		ui->cbox_DIqidonganniu->setEnabled(true);
		ui->cbox_DIjiting->setEnabled(true);
		ui->cbox_DIlalianlawan->setEnabled(true);
		ui->cbox_DObujindianjimaichong->setEnabled(true);
		ui->cbox_DOchongkong->setEnabled(true);
		ui->cbox_DOtuoji->setEnabled(true);
		ui->cbox_DOchufapaizhao->setEnabled(true);

		globalStruct.monitorZMotionMonitorThread.setRunning(false);
	}
	else
	{
		ui->cbox_DIqidonganniu->setChecked(false);
		ui->cbox_DIjiting->setChecked(false);
		ui->cbox_DIlalianlawan->setChecked(false);
		ui->cbox_DObujindianjimaichong->setChecked(false);
		ui->cbox_DOchongkong->setChecked(false);
		ui->cbox_DOtuoji->setChecked(false);
		ui->cbox_DOchufapaizhao->setChecked(false);

		ui->cbox_DIqidonganniu->setEnabled(false);
		ui->cbox_DIjiting->setEnabled(false);
		ui->cbox_DIlalianlawan->setEnabled(false);
		ui->cbox_DObujindianjimaichong->setEnabled(false);
		ui->cbox_DOchongkong->setEnabled(false);
		ui->cbox_DOtuoji->setEnabled(false);
		ui->cbox_DOchufapaizhao->setEnabled(false);

		globalStruct.monitorZMotionMonitorThread.setRunning(true);
	}
}

void DlgProductSet::cBox_takeCamera1Pictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.takeWork1Pictures = ui->cBox_takeCamera1Pictures->isChecked();
}

void DlgProductSet::cBox_takeCamera2Pictures_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.takeWork2Pictures = ui->cBox_takeCamera2Pictures->isChecked();
}

void DlgProductSet::cbox_qiyongerxiangji_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.qiyongerxiangji = ui->cbox_qiyongerxiangji->isChecked();
}

void DlgProductSet::cbox_qiyongyundongkongzhiqi_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.qiyongyundongkongzhiqi = ui->cbox_qiyongyundongkongzhiqi->isChecked();
}

void DlgProductSet::cbox_yundongkongzhiqichonglian_checked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.yundongkongzhiqichonglian = ui->cbox_yundongkongzhiqichonglian->isChecked();
}

void DlgProductSet::btn_shedingladaichangdu_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_shedingladaichangdu->setText(value);
		globalStructSetConfig.shedingladaichangdu = value.toDouble();
	}
}

void DlgProductSet::btn_xiangjichufachangdu_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_xiangjichufachangdu->setText(value);
		globalStructSetConfig.xiangjichufachangdu = value.toDouble();
		bool isSet = globalStruct.zmotion.setModbus(0, 4, value.toDouble());

		if (!isSet)
		{
			QMessageBox::warning(this, "警告", "设定拉袋长度失败!");
		}
	}
}

void DlgProductSet::btn_jishuqingling_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.chongkongjishu = 0;
	ui->btn_chongkongjishu->setText("0");
}

void DlgProductSet::btn_changduqingling_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	globalStructSetConfig.dangqianchangdu = 0;
	ui->btn_dangqianchangdu->setText("0");
}

void DlgProductSet::btn_shoudongladai_pressed()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto setConfig = globalStruct.setConfig;
	auto value = setConfig.meizhuanmaichongshu / setConfig.shedingzhouchang;

	// 启动电机
	auto isAxisType = globalStruct.zmotion.setAxisType(0, 1);
	double unit = value;
	auto isAxisPulse = globalStruct.zmotion.setAxisPulse(0, unit);
	double acc = setConfig.jiajiansushijian;
	auto isAxisAcc = globalStruct.zmotion.setAxisAcc(0, acc);
	auto isAxisDec = globalStruct.zmotion.setAxisDec(0, acc);
	double speed = setConfig.shoudongsudu;
	auto isAxisRunSpeed = globalStruct.zmotion.setAxisRunSpeed(0, speed);
	auto isAxisRun = globalStruct.zmotion.setAxisRun(0, -1);

	if (!isAxisType || !isAxisPulse || !isAxisAcc || !isAxisDec || !isAxisRunSpeed || !isAxisRun)
	{
		QMessageBox::warning(this, "警告", "电机参数设置失败");
	}
}

void DlgProductSet::btn_shoudongladai_released()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	// 停止电机
	bool isStop = globalStruct.zmotion.stopAllAxis();

	if (!isStop)
	{
		QMessageBox::warning(this, "警告", "手动拉带取消失败!");
	}
}

void DlgProductSet::btn_shoudongchongkong_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	bool isSet = globalStruct.zmotion.SetIOOut(2, ControlLines::chongkongOUT, true, 1000);

	if (!isSet)
	{
		QMessageBox::warning(this, "警告", "手动冲孔失败!");
	}
}

void DlgProductSet::btn_tuoji_clicked()
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();

	bool isGetTuoJiOut = false;
	bool tuojiState = globalStruct.zmotion.getIOOut(ControlLines::tuojiOut, isGetTuoJiOut);
	bool isSuccess = globalStruct.zmotion.setIOOut(ControlLines::tuojiOut, !tuojiState);
	if (!isSuccess)
	{
		QMessageBox::warning(this, "警告", "设置脱机失败!");
	}

	if (!tuojiState)
	{
		ui->btn_tuoji->setText("已脱机...");
		ui->btn_tuoji->setStyleSheet(
			"QPushButton {"
			"border-color: #388E3C;"
			"background-color: #4CAF50;"
			"color: #fff;"
			"font-weight: bold;"
			"}"
		);
	}
	else
	{
		ui->btn_tuoji->setText("脱机");
		ui->btn_tuoji->setStyleSheet(
			"QPushButton {"
			"padding: 6px 14px;"
			"border: 2px solid #CCC;"
			"border-radius: 4px;"
			"background-color: red;"
			"color: #444;"
			"}"
		);
	}
}



void DlgProductSet::btn_xiangjichufa_clicked()
{

}

void DlgProductSet::btn_shoudongsudu_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_shoudongsudu->setText(value);
		globalStructSetConfig.shoudongsudu = value.toDouble();
	}
}

void DlgProductSet::btn_meizhuanmaichongshu_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_meizhuanmaichongshu->setText(value);
		globalStructSetConfig.meizhuanmaichongshu = value.toDouble();
	}
}

void DlgProductSet::btn_zidongladaisudu_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_zidongladaisudu->setText(value);
		globalStructSetConfig.zidongladaisudu = value.toDouble();
	}
}

void DlgProductSet::btn_shedingzhouchang_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_shedingzhouchang->setText(value);
		globalStructSetConfig.shedingzhouchang = value.toDouble();
	}
}

void DlgProductSet::btn_yanshichongkong_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_yanshichongkong->setText(value);
		globalStructSetConfig.yanshichongkong = value.toDouble();
	}
}

void DlgProductSet::btn_chongkongshijian_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_chongkongshijian->setText(value);
		globalStructSetConfig.chongkongshijian = value.toDouble();
	}
}

void DlgProductSet::btn_yanshiziqi_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_yanshiziqi->setText(value);
		globalStructSetConfig.yanshiziqi = value.toDouble();
	}
}

void DlgProductSet::btn_jiajiansushijian_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_jiajiansushijian->setText(value);
		globalStructSetConfig.jiajiansushijian = value.toDouble();
	}
}

void DlgProductSet::cbox_DIqidonganniu_clicked(bool isChecked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	if (isDebugIO)
	{
		auto isSuccess = globalStruct.zmotion.setIOOut(ControlLines::qidonganniuIn, isChecked);
	}
}

void DlgProductSet::cbox_DIjiting_clicked(bool isChecked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	if (isDebugIO)
	{
		auto isSuccess = globalStruct.zmotion.setIOOut(ControlLines::jitingIn, isChecked);
	}
}

void DlgProductSet::cbox_DIlalianlawan_clicked(bool isChecked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	if (isDebugIO)
	{
		auto isSuccess = globalStruct.zmotion.setIOOut(ControlLines::lalianlawanIn, isChecked);
	}
}

void DlgProductSet::cbox_DObujindianjimaichong_clicked(bool isChecked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	if (isDebugIO)
	{
		auto isSuccess = globalStruct.zmotion.setIOOut(ControlLines::bujindianjimaichongOut, isChecked);
	}
}

void DlgProductSet::cbox_DOchongkong_clicked(bool isChecked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	if (isDebugIO)
	{
		auto isSuccess = globalStruct.zmotion.setIOOut(ControlLines::chongkongOUT, isChecked);
	}
}

void DlgProductSet::cbox_DOtuoji_clicked(bool isChecked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	if (isDebugIO)
	{
		auto isSuccess = globalStruct.zmotion.setIOOut(ControlLines::tuojiOut, isChecked);
	}
}

void DlgProductSet::cbox_DOchufapaizhao_clicked(bool isChecked)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	if (isDebugIO)
	{
		auto isSuccess = globalStruct.zmotion.setIOOut(ControlLines::chufapaizhaoOUT, isChecked);
	}
}

void DlgProductSet::btn_setqidonganniu_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_setqidonganniu->setText(value);
		globalStructSetConfig.qidonganniuIN = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		auto indicesDI = DIFindAllDuplicateIndices();
		setDIErrorInfo(indicesDI);
	}
}

void DlgProductSet::btn_setlalianlawan_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_setlalianlawan->setText(value);
		globalStructSetConfig.lalianlawanIN = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		auto indicesDI = DIFindAllDuplicateIndices();
		setDIErrorInfo(indicesDI);
	}
}

void DlgProductSet::btn_setjiting_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_setjiting->setText(value);
		globalStructSetConfig.jitingIN = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		auto indicesDI = DIFindAllDuplicateIndices();
		setDIErrorInfo(indicesDI);
	}
}

void DlgProductSet::btn_setbujindianjimaichong_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_setbujindianjimaichong->setText(value);
		globalStructSetConfig.bujindianjimaichongOUT = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		auto indicesDI = DIFindAllDuplicateIndices();
		setDIErrorInfo(indicesDI);
	}
}

void DlgProductSet::btn_setchongkong_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_setchongkong->setText(value);
		globalStructSetConfig.chongkongOUT = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		auto indicesDI = DIFindAllDuplicateIndices();
		setDIErrorInfo(indicesDI);
	}
}

void DlgProductSet::btn_settuoji_clicked()
{
	auto& globalStructSetConfig = GlobalStructDataZipper::getInstance().setConfig;
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0)
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		ui->btn_settuoji->setText(value);
		globalStructSetConfig.tuojiOUT = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		auto indicesDI = DIFindAllDuplicateIndices();
		setDIErrorInfo(indicesDI);
	}
}

void DlgProductSet::tabWidget_indexChanged(int index)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	auto& _isUpdateMonitorInfo = GlobalStructDataZipper::getInstance()._isUpdateMonitorInfo;
	switch (index) {
	case 0:
		_isUpdateMonitorInfo = false;
		globalStruct.monitorZMotionMonitorThread.setRunning(false);
		break;
	case 1:
		_isUpdateMonitorInfo = false;
		globalStruct.monitorZMotionMonitorThread.setRunning(false);
		break;
	case 2:
		_isUpdateMonitorInfo = true;
		globalStruct.monitorZMotionMonitorThread.setRunning(true);
		break;
	case 3:
		_isUpdateMonitorInfo = false;
		globalStruct.monitorZMotionMonitorThread.setRunning(false);
		break;
	default:
		_isUpdateMonitorInfo = false;
		globalStruct.monitorZMotionMonitorThread.setRunning(false);
		break;
	}
}

void DlgProductSet::monitorInPutSignal(size_t index, bool state)
{
	if (isDebugIO == false)
	{
		switch (index)
		{
		case ControlLines::qidonganniuIn: // 启动按钮
			if (state)
			{
				ui->cbox_DIqidonganniu->setChecked(true);
			}
			else
			{
				ui->cbox_DIqidonganniu->setChecked(false);
			}
			break;
		case ControlLines::jitingIn: // 急停按钮
			if (state)
			{
				ui->cbox_DIjiting->setChecked(true);
			}
			else
			{
				ui->cbox_DIjiting->setChecked(false);
			}
			break;
		case ControlLines::lalianlawanIn: // 拉链拉完按钮
			if (state)
			{
				ui->cbox_DIlalianlawan->setChecked(true);
			}
			else
			{
				ui->cbox_DIlalianlawan->setChecked(false);
			}
			break;
		}

	}

}

void DlgProductSet::monitorOutPutSignal(size_t index, bool state)
{
	if (isDebugIO == false)
	{
		switch (index)
		{
		case ControlLines::bujindianjimaichongOut: // 步进电机脉冲按钮
			if (state)
			{
				ui->cbox_DObujindianjimaichong->setChecked(true);
			}
			else
			{
				ui->cbox_DObujindianjimaichong->setChecked(false);
			}
			break;
		case ControlLines::chongkongOUT: // 冲孔按钮
			if (state)
			{
				ui->cbox_DOchongkong->setChecked(true);
			}
			else
			{
				ui->cbox_DOchongkong->setChecked(false);
			}
			break;
		case ControlLines::tuojiOut: // 脱机按钮
			if (state)
			{
				ui->cbox_DOtuoji->setChecked(true);
			}
			else
			{
				ui->cbox_DOtuoji->setChecked(false);
			}
			break;
		case ControlLines::chufapaizhaoOUT: // 触发拍照按钮
			if (state)
			{
				ui->cbox_DOchufapaizhao->setChecked(true);
			}
			else
			{
				ui->cbox_DOchufapaizhao->setChecked(false);
			}
			break;
		}
	}
}




