#include "DlgProductSet.h"

#include "GlobalStruct.hpp"
#include <NumberKeyboard.h>
#include <QMessageBox>
#include "Utilty.hpp"
#include"rqw_MonitorMotionIO.hpp"

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
	_monitorZmotion->destroyThread();
	delete ui;
}

std::vector<std::vector<int>> DlgProductSetSmartCroppingOfBags::DOFindAllDuplicateIndices()
{
	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	std::vector<int> values = {
		setConfig.chuiqiOUT,
		setConfig.baojinghongdengOUT,
		setConfig.yadaiOUT,
		setConfig.tifeiOUT
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

void DlgProductSetSmartCroppingOfBags::setDOErrorInfo(const std::vector<std::vector<int>>& index)
{
	ui->lb_chuiqi->clear();
	ui->lb_baojinghongdeng->clear();
	ui->lb_yadai->clear();
	ui->lb_tifei->clear();

	for (const auto& classic : index)
	{
		for (const auto& item : classic)
		{
			setDOErrorInfo(item);
		}
	}
}

void DlgProductSetSmartCroppingOfBags::setDOErrorInfo(int index)
{
	QString text = "重复数值";
	switch (index)
	{
	case 0:
		ui->lb_chuiqi->setText(text);
		break;
	case 1:
		ui->lb_baojinghongdeng->setText(text);
		break;
	case 2:
		ui->lb_yadai->setText(text);
		break;
	case 3:
		ui->lb_tifei->setText(text);
		break;
	}
}

void DlgProductSetSmartCroppingOfBags::changeBagType(int index)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& globalStructSetConfig = globalStruct.setConfig;
	bool isSet = false;
	// 颜色袋
	if (index)
	{
		if (globalStruct.camera1)
		{
			isSet = globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.xiangjizengyi1));
		}
	}
	// 白色袋
	else if (index == 0)
	{
		if (globalStructSetConfig.isxiangjizengyi1)
		{
			if (globalStruct.camera1)
			{
				isSet = globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.xiangjizengyi1));
			}
		}
		else
		{
			if (globalStruct.camera1)
			{
				isSet = globalStruct.camera1->setGain(0);
			}
		}
	}
	if (!isSet)
	{
		QMessageBox::warning(this, "警告", "设置相机增益失败!");
	}
}

void DlgProductSetSmartCroppingOfBags::showEvent(QShowEvent* showEvent)
{
	QDialog::showEvent(showEvent);
	_monitorZmotion->setMonitorObject(GlobalStructDataSmartCroppingOfBags::getInstance().zMotion);
	_monitorZmotion->setRunning(true);
	if (ui->tabWidget->currentIndex()==2)
	{
		auto& _isUpdateMonitorInfo = GlobalStructThreadSmartCroppingOfBags::getInstance()._isUpdateMonitorInfo;
		_isUpdateMonitorInfo = true;
	}
}

void DlgProductSetSmartCroppingOfBags::build_ui()
{
	read_config();
	auto indicesDO = DOFindAllDuplicateIndices();
	setDOErrorInfo(indicesDO);

	ui->tabWidget->setCurrentIndex(0);

	_monitorZmotion = std::make_unique<rw::rqw::MonitorZMotionIOStateThread>();
	_monitorZmotion->setRunning(false);
	_monitorZmotion->setMonitorFrequency(100);
	_monitorZmotion->setMonitorIList({ControlLines::qiedaoIn});
	_monitorZmotion->setMonitorOList({ ControlLines::baojinghongdengOUT,ControlLines ::chuiqiOut,ControlLines ::tifeiOut,ControlLines ::yadaiOut});
	_monitorZmotion->start();
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
	ui->ckb_yundongkongzhiqichonglian->setChecked(globalConfig.yundongkongzhiqichonglian);
	ui->btn_jiange->setText(QString::number(globalConfig.jiange));

	// 1相机参数
	ui->btn_pingjunmaichong1->setText(QString::number(globalConfig.pingjunmaichong1));
	ui->btn_maichongxinhao1->setText(QString::number(globalConfig.maichongxinhao1));
	ui->btn_hanggao1->setText(QString::number(globalConfig.hanggao1));
	ui->btn_daichang1->setText(QString::number(globalConfig.daichang1));
	ui->btn_daichangxishu1->setText(QString::number(globalConfig.daichangxishu1));
	ui->btn_guasijuli1->setText(QString::number(globalConfig.guasijuli1));
	ui->btn_zuixiaochutu1->setText(QString::number(globalConfig.zuixiaochutu1));
	ui->btn_zuidachutu1->setText(QString::number(globalConfig.zuidachutu1));
	ui->btn_baisedailiangdufanweiMin1->setText(QString::number(globalConfig.baisedailiangdufanweimin1));
	ui->btn_baisedailiangdufanweiMax1->setText(QString::number(globalConfig.baisedailiangdufanweimax1));

	ui->btn_daokoudaoxiangjijuli1->setText(QString::number(globalConfig.daokoudaoxiangjiluli1));
	ui->btn_xiangjibaoguang1->setText(QString::number(globalConfig.xiangjibaoguang1));
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

	// IO接口设置参数
	// 输入
	ui->btn_qiedao->setText(QString::number(globalConfig.qiedaoIN));
	ControlLines::qiedaoIn = globalConfig.qiedaoIN;
	// 输出
	ui->btn_chuiqi->setText(QString::number(globalConfig.chuiqiOUT));
	ControlLines::chuiqiOut = globalConfig.chuiqiOUT;

	ui->btn_baojinghongdeng->setText(QString::number(globalConfig.baojinghongdengOUT));
	ControlLines::baojinghongdengOUT = globalConfig.baojinghongdengOUT;

	ui->btn_yadai->setText(QString::number(globalConfig.yadaiOUT));
	ControlLines::yadaiOut = globalConfig.yadaiOUT;

	ui->btn_tifei->setText(QString::number(globalConfig.tifeiOUT));
	ControlLines::tifeiOut = globalConfig.tifeiOUT;

}

void DlgProductSetSmartCroppingOfBags::build_connect()
{
	// 连接槽函数
	// 按钮点击信号连接
	QObject::connect(ui->btn_zidongpingbifanwei, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_zidongpingbifanwei_clicked);
	QObject::connect(ui->btn_jiange, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_jiange_clicked);
	QObject::connect(ui->btn_daichangxishu1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_daichangxishu1_clicked);
	QObject::connect(ui->btn_guasijuli1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_guasijuli1_clicked);
	QObject::connect(ui->btn_zuixiaochutu1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_zuixiaodaichang1_clicked);
	QObject::connect(ui->btn_zuidachutu1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_zuidadaichang1_clicked);
	QObject::connect(ui->btn_baisedailiangdufanweiMin1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_baisedailiangdufanweiMin1_clicked);
	QObject::connect(ui->btn_baisedailiangdufanweiMax1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_baisedailiangdufanweiMax1_clicked);
	QObject::connect(ui->btn_daokoudaoxiangjijuli1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_daokoudaoxiangjijuli1_clicked);
	QObject::connect(ui->btn_tifeiyanshi1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_tifeiyanshi1_clicked);
	QObject::connect(ui->btn_baojingyanshi1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_baojingyanshi1_clicked); 
		QObject::connect(ui->btn_baojingshijian1, &QPushButton::clicked,
			this, &DlgProductSetSmartCroppingOfBags::btn_baojingshijian1_clicked);
	QObject::connect(ui->btn_tifeishijian1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_tifeishijian1_clicked);
	QObject::connect(ui->btn_chuiqiyanshi1, &QPushButton::clicked, 
		this, &DlgProductSetSmartCroppingOfBags::btn_chuiqiyanshi1_clicked);
	QObject::connect(ui->btn_dudaiyanshi1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_dudaiyanshi1_clicked);
	QObject::connect(ui->btn_chuiqishijian1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_chuiqishijian1_clicked);
	QObject::connect(ui->btn_dudaishijian1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_dudaishijian1_clicked);
	QObject::connect(ui->btn_maichongxishu1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_maichongxishu1_clicked);
	QObject::connect(ui->btn_xiangjizengyi1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_xiangjizengyi1_clicked);
	QObject::connect(ui->btn_houfenpinqi1, &QPushButton::clicked, 
		this, &DlgProductSetSmartCroppingOfBags::btn_houfenpinqi1_clicked);
	QObject::connect(ui->btn_chengfaqi1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_chengfaqi1_clicked);
	QObject::connect(ui->btn_qiedaoxianshangpingbi1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_qiedaoxianshangpingbi1_clicked);
	QObject::connect(ui->btn_qiedaoxianxiapingbi1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_qiedaoxianxiapingbi1_clicked);
	QObject::connect(ui->btn_yansedailiangdufanweiMin1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_yansedailiangdufanweiMin1_clicked);
	QObject::connect(ui->btn_yansedailiangdufanweiMax1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_yansedailiangdufanweiMax1_clicked);
	QObject::connect(ui->btn_xiangjibaoguang1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_xiangjibaoguang1_clicked);

	// 复选框勾选信号连接
	QObject::connect(ui->ckb_xiaopodong, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_xiaopodong_checked);
	QObject::connect(ui->ckb_tiqiantifei, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_tiqiantifei_checked);
	QObject::connect(ui->ckb_xiangjitiaoshi, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_xiangjitiaoshi_checked);
	QObject::connect(ui->ckb_qiyonger, &QCheckBox::clicked, 
		this, &DlgProductSetSmartCroppingOfBags::ckb_qiyonger_checked);
	QObject::connect(ui->ckb_yundongkongzhiqichonglian, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_yundongkongzhiqichonglian_checked);
	QObject::connect(ui->ckb_xiangjizengyi, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_xiangjizengyi_checked);

	// 连接关闭按钮
	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::pbtn_close_clicked);

	QObject::connect(ui->btn_qiedao, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_qiedao_clicked);
	QObject::connect(ui->btn_chuiqi, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_chuiqi_clicked);
	QObject::connect(ui->btn_baojinghongdeng, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_baojinghongdeng_clicked);
	QObject::connect(ui->btn_yadai, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_yadai_clicked);
	QObject::connect(ui->btn_tifei, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_tifei_clicked);

	QObject::connect(ui->ckb_baojinghongdeng, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_baojinghongdeng_checked);
	QObject::connect(ui->ckb_qiedao, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_qiedao_checked);
	QObject::connect(ui->ckb_chuiqi, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_chuiqi_checked);
	QObject::connect(ui->ckb_yadai, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_yadai_checked);
	QObject::connect(ui->ckb_tifei, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_tifei_checked);
	QObject::connect(ui->ckb_debugIO, &QCheckBox::clicked,
		this, &DlgProductSetSmartCroppingOfBags::ckb_debugIO_checked);

	QObject::connect(_monitorZmotion.get(), &rw::rqw::MonitorZMotionIOStateThread::DIState,
		this, &DlgProductSetSmartCroppingOfBags::onDIState);
	QObject::connect(_monitorZmotion.get(), &rw::rqw::MonitorZMotionIOStateThread::DOState,
		this, &DlgProductSetSmartCroppingOfBags::onDOState);

	QObject::connect(ui->tabWidget, &QTabWidget::currentChanged,
		this, &DlgProductSetSmartCroppingOfBags::tabWidget_indexChanged);
	QObject::connect(ui->btn_daichang1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_daichang1_clicked);
}

void DlgProductSetSmartCroppingOfBags::onUpdateMonitorRunningStateInfo(MonitorRunningStateInfo info)
{
	if (info.isGetCurrentPulse)
	{
		ui->btn_maichongxinhao1->setText(QString::number(info.currentPulse, 'f', 2));
	}
	if (info.isGetAveragePixelBag)
	{
		ui->btn_pinjunxiangsudangliangdaichang->setText(QString::number(info.averagePixelBag, 'f', 2));

	}
	if (info.isGetAveragePulseBag)
	{
		ui->btn_pingjunmaichongdaichang->setText(QString::number(info.averagePulseBag, 'f', 2));

	}
	if (info.isGetAveragePulse)
	{
		ui->btn_pingjunmaichong1->setText(QString::number(info.averagePulse, 'f', 2));

	}
	if (info.isGetLineHeight)
	{
		ui->btn_hanggao1->setText(QString::number(info.lineHeight, 'f', 2));
	}
}

void DlgProductSetSmartCroppingOfBags::pbtn_close_clicked()
{
	auto& _isUpdateMonitorInfo = GlobalStructThreadSmartCroppingOfBags::getInstance()._isUpdateMonitorInfo;
	_isUpdateMonitorInfo = false;
	_monitorZmotion->setRunning(false);
	auto& GlobalStructData = GlobalStructDataSmartCroppingOfBags::getInstance();
	GlobalStructData.saveDlgProductSetConfig();
	this->close();
}

void DlgProductSetSmartCroppingOfBags::btn_zidongpingbifanwei_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_zidongpingbifanwei->setText(value);
		globalStructSetConfig.zidongpingbifanwei = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_jiange_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_jiange->setText(value);
		globalStructSetConfig.jiange = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_daichangxishu1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_daichangxishu1->setText(value);
		globalStructSetConfig.daichangxishu1 = value.toDouble();

		GlobalStructThreadSmartCroppingOfBags::getInstance()._detachUtiltyThreadSmartCroppingOfBags->runningStatePixelParaChange = true;
	}
}

void DlgProductSetSmartCroppingOfBags::btn_guasijuli1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_guasijuli1->setText(value);
		globalStructSetConfig.guasijuli1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_zuixiaodaichang1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_zuixiaochutu1->setText(value);
		globalStructSetConfig.zuixiaochutu1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_zuidadaichang1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_zuidachutu1->setText(value);
		globalStructSetConfig.zuidachutu1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_baisedailiangdufanweiMin1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_baisedailiangdufanweiMin1->setText(value);
		globalStructSetConfig.baisedailiangdufanweimin1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_baisedailiangdufanweiMax1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_baisedailiangdufanweiMax1->setText(value);
		globalStructSetConfig.baisedailiangdufanweimax1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_daokoudaoxiangjijuli1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_daokoudaoxiangjijuli1->setText(value);
		globalStructSetConfig.daokoudaoxiangjiluli1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_xiangjibaoguang1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 500)
		{
			QMessageBox::warning(this, "提示", "请输入大于0小于500的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		globalStruct.camera1->setExposureTime(value.toDouble());
		ui->btn_xiangjibaoguang1->setText(value);
		globalStructSetConfig.xiangjibaoguang1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_tifeiyanshi1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_tifeiyanshi1->setText(value);
		globalStructSetConfig.tifeiyanshi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_baojingyanshi1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_baojingyanshi1->setText(value);
		globalStructSetConfig.baojingyanshi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_baojingshijian1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_baojingshijian1->setText(value);
		globalStructSetConfig.baojingshijian1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_tifeishijian1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_tifeishijian1->setText(value);
		globalStructSetConfig.tifeishijian1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_chuiqiyanshi1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_chuiqiyanshi1->setText(value);
		globalStructSetConfig.chuiqiyanshi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_dudaiyanshi1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_dudaiyanshi1->setText(value);
		globalStructSetConfig.dudaiyanshi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_chuiqishijian1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_chuiqishijian1->setText(value);
		globalStructSetConfig.chuiqishijian1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_dudaishijian1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_dudaishijian1->setText(value);
		globalStructSetConfig.dudaishijian1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_maichongxishu1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_maichongxishu1->setText(value);
		globalStructSetConfig.maichongxishu1 = value.toDouble();
		GlobalStructThreadSmartCroppingOfBags::getInstance()._detachUtiltyThreadSmartCroppingOfBags->runningStatePulseParaChange = true;
	}
}

void DlgProductSetSmartCroppingOfBags::btn_xiangjizengyi1_clicked()
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
		auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		auto& generalConfig = globalStruct.generalConfig;

		bool isSet = false;
		// 颜色袋
		if (generalConfig.daizizhonglei == 1)
		{
			if (globalStruct.camera1)
			{
				isSet = globalStruct.camera1->setGain(value.toDouble());
			}
		}
		// 白色袋
		else if (generalConfig.daizizhonglei == 0)
		{
			if (globalStructSetConfig.isxiangjizengyi1)
			{
				if (globalStruct.camera1)
				{
					isSet = globalStruct.camera1->setGain(value.toDouble());
				}
			}
			else
			{
				if (globalStruct.camera1)
				{
					isSet = globalStruct.camera1->setGain(0);
				}
			}
		}
		ui->btn_xiangjizengyi1->setText(value);
		globalStructSetConfig.xiangjizengyi1 = value.toDouble();

		if (!isSet)
		{
			QMessageBox::warning(this, "警告", "设置相机增益失败!");
		}
	}
}

void DlgProductSetSmartCroppingOfBags::btn_houfenpinqi1_clicked()
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
		auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		globalStruct.camera1->setPostDivider(value.toInt());
		ui->btn_houfenpinqi1->setText(value);
		globalStructSetConfig.houfenpinqi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_chengfaqi1_clicked()
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
		auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
		auto& globalStructSetConfig = globalStruct.setConfig;
		globalStruct.camera1->setMultiplier(value.toInt());
		ui->btn_chengfaqi1->setText(value);
		globalStructSetConfig.chengfaqi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_qiedaoxianshangpingbi1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_qiedaoxianshangpingbi1->setText(value);
		globalStructSetConfig.qiedaoxianshangpingbi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_qiedaoxianxiapingbi1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_qiedaoxianxiapingbi1->setText(value);
		globalStructSetConfig.qiedaoxianxiapingbi1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_yansedailiangdufanweiMin1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_yansedailiangdufanweiMin1->setText(value);
		globalStructSetConfig.yansedailiangdufanweimin1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_yansedailiangdufanweiMax1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_yansedailiangdufanweiMax1->setText(value);
		globalStructSetConfig.yansedailiangdufanweimax1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_daichang1_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_daichang1->setText(value);
		globalStructSetConfig.daichang1 = value.toDouble();

		if (GlobalStructDataSmartCroppingOfBags::getInstance().removeState == RemoveState::SmartCrop)
		{
			auto lineHeight = globalStructSetConfig.daichang1 / globalStructSetConfig.daichangxishu1;
			GlobalStructDataSmartCroppingOfBags::getInstance().camera1->setLineHeight(lineHeight);
		}
	}
}

void DlgProductSetSmartCroppingOfBags::ckb_xiaopodong_checked()
{
	auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	globalStructSetConfig.xiaopodong = ui->ckb_xiaopodong->isChecked();
}

void DlgProductSetSmartCroppingOfBags::ckb_tiqiantifei_checked()
{
	auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	globalStructSetConfig.tiqiantifei = ui->ckb_tiqiantifei->isChecked();
}

void DlgProductSetSmartCroppingOfBags::ckb_xiangjitiaoshi_checked()
{
	auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	globalStructSetConfig.xiangjitiaoshi = ui->ckb_xiangjitiaoshi->isChecked();
}

void DlgProductSetSmartCroppingOfBags::ckb_qiyonger_checked()
{
	auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	globalStructSetConfig.qiyonger = ui->ckb_qiyonger->isChecked();
}

void DlgProductSetSmartCroppingOfBags::ckb_yundongkongzhiqichonglian_checked()
{
	auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	globalStructSetConfig.yundongkongzhiqichonglian = ui->ckb_yundongkongzhiqichonglian->isChecked();
}

void DlgProductSetSmartCroppingOfBags::ckb_xiangjizengyi_checked(bool ischecked)
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	auto& generalConfig = globalStruct.generalConfig;
	auto& globalStructSetConfig = globalStruct.setConfig;
	globalStructSetConfig.isxiangjizengyi1 = ischecked;

	bool isSet = false;
	if (ischecked)
	{
		if (globalStruct.camera1)
		{
			isSet = globalStruct.camera1->setGain(static_cast<size_t>(globalStructSetConfig.xiangjizengyi1));
		}
	}
	else
	{
		if (globalStruct.camera1)
		{
			isSet = globalStruct.camera1->setGain(0);
		}
	}
	if (!isSet)
	{
		QMessageBox::warning(this, "警告", "设置增益失败!");
	}
}

void DlgProductSetSmartCroppingOfBags::btn_qiedao_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_qiedao->setText(value);
		globalStructSetConfig.qiedaoIN = value.toDouble();
		ControlLines::qiedaoIn = value.toUInt();

	}
}

void DlgProductSetSmartCroppingOfBags::btn_chuiqi_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_chuiqi->setText(value);
		globalStructSetConfig.chuiqiOUT = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		ControlLines::chuiqiOut = value.toUInt();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_baojinghongdeng_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_baojinghongdeng->setText(value);
		globalStructSetConfig.baojinghongdengOUT = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		ControlLines::baojinghongdengOUT = value.toUInt();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_yadai_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_yadai->setText(value);
		globalStructSetConfig.yadaiOUT = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		ControlLines::yadaiOut = value.toUInt();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_tifei_clicked()
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_tifei->setText(value);
		globalStructSetConfig.tifeiOUT = value.toDouble();
		auto indicesDO = DOFindAllDuplicateIndices();
		setDOErrorInfo(indicesDO);
		ControlLines::tifeiOut = value.toUInt();
	}
}

void DlgProductSetSmartCroppingOfBags::ckb_debugIO_checked(bool ischecked)
{
	isDebugIO = ischecked;
	if (isDebugIO)
	{
		ui->ckb_qiedao->setChecked(false);
		ui->ckb_chuiqi->setChecked(false);
		ui->ckb_baojinghongdeng->setChecked(false);
		ui->ckb_yadai->setChecked(false);
		ui->ckb_tifei->setChecked(false);

		ui->ckb_qiedao->setEnabled(true);
		ui->ckb_chuiqi->setEnabled(true);
		ui->ckb_baojinghongdeng->setEnabled(true);
		ui->ckb_yadai->setEnabled(true);
		ui->ckb_tifei->setEnabled(true);
	}
	else
	{
		ui->ckb_qiedao->setChecked(false);
		ui->ckb_chuiqi->setChecked(false);
		ui->ckb_baojinghongdeng->setChecked(false);
		ui->ckb_yadai->setChecked(false);
		ui->ckb_tifei->setChecked(false);

		ui->ckb_qiedao->setEnabled(false);
		ui->ckb_chuiqi->setEnabled(false);
		ui->ckb_baojinghongdeng->setEnabled(false);
		ui->ckb_yadai->setEnabled(false);
		ui->ckb_tifei->setEnabled(false);
	}
	_monitorZmotion->setRunning(!ischecked);
}

void DlgProductSetSmartCroppingOfBags::ckb_qiedao_checked(bool ischecked)
{
	return;
}

void DlgProductSetSmartCroppingOfBags::ckb_chuiqi_checked(bool ischecked)
{
	if (!isDebugIO)
	{
		return;
	}
	auto& ZMotion = GlobalStructDataSmartCroppingOfBags::getInstance().zMotion;
	ZMotion.setIOOut(ControlLines::chuiqiOut, ischecked);
}

void DlgProductSetSmartCroppingOfBags::ckb_baojinghongdeng_checked(bool ischecked)
{
	if (!isDebugIO)
	{
		return;
	}
	auto& ZMotion = GlobalStructDataSmartCroppingOfBags::getInstance().zMotion;
	ZMotion.setIOOut(ControlLines::baojinghongdengOUT, ischecked);
}

void DlgProductSetSmartCroppingOfBags::ckb_yadai_checked(bool ischecked)
{
	if (!isDebugIO)
	{
		return;
	}
	auto& ZMotion = GlobalStructDataSmartCroppingOfBags::getInstance().zMotion;
	ZMotion.setIOOut(ControlLines::yadaiOut, ischecked);
}

void DlgProductSetSmartCroppingOfBags::ckb_tifei_checked(bool ischecked)
{
	if (!isDebugIO)
	{
		return;
	}
	auto& ZMotion = GlobalStructDataSmartCroppingOfBags::getInstance().zMotion;
	ZMotion.setIOOut(ControlLines::tifeiOut, ischecked);
}

void DlgProductSetSmartCroppingOfBags::tabWidget_indexChanged(int index)
{
	auto& _isUpdateMonitorInfo = GlobalStructThreadSmartCroppingOfBags::getInstance()._isUpdateMonitorInfo;
	switch (index) {
	case 0:
		_isUpdateMonitorInfo = false;
		break;
	case 1:
		_isUpdateMonitorInfo = false;
		break;
	case 2:
		_isUpdateMonitorInfo = true;
		break;
	case 3:
		_isUpdateMonitorInfo = false;
		break;
	default:
		_isUpdateMonitorInfo = false;
		break;
	}
}

void DlgProductSetSmartCroppingOfBags::onDIState(size_t index, bool state)
{
	if (index == ControlLines::qiedaoIn)
	{
		ui->ckb_qiedao->setChecked(state);
	}

}

void DlgProductSetSmartCroppingOfBags::onDOState(size_t index, bool state)
{
	if (index == ControlLines::baojinghongdengOUT)
	{
		ui->ckb_baojinghongdeng->setChecked(state);
	}
	else if (index == ControlLines::chuiqiOut)
	{
		ui->ckb_chuiqi->setChecked(state);
	}
	else if (index == ControlLines::tifeiOut)
	{
		ui->ckb_tifei->setChecked(state);
	}
	else if (index == ControlLines::yadaiOut)
	{
		ui->ckb_yadai->setChecked(state);
	}
}


