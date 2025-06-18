#include "DlgProductSet.h"

#include "GlobalStruct.hpp"
#include <NumberKeyboard.h>
#include <QMessageBox>

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
	ui->ckb_yundongkongzhiqichonglian->setChecked(globalConfig.yundongkongzhiqichonglian);
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
}

void DlgProductSetSmartCroppingOfBags::build_connect()
{
	// 连接槽函数
	// 按钮点击信号连接
	QObject::connect(ui->btn_zidongpingbifanwei, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_zidongpingbifanwei_clicked);
	QObject::connect(ui->btn_jiange, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_jiange_clicked);
	QObject::connect(ui->btn_pingjunmaichong1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_pingjunmaichong1_clicked);
	QObject::connect(ui->btn_maichongxinhao1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_maichongxinhao1_clicked);
	QObject::connect(ui->btn_hanggao1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_hanggao1_clicked);
	QObject::connect(ui->btn_daichang1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_daichang1_clicked);
	QObject::connect(ui->btn_daichangxishu1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_daichangxishu1_clicked);
	QObject::connect(ui->btn_guasijuli1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_guasijuli1_clicked);
	QObject::connect(ui->btn_zuixiaodaichang1, &QPushButton::clicked,
		this, &DlgProductSetSmartCroppingOfBags::btn_zuixiaodaichang1_clicked);
	QObject::connect(ui->btn_zuidadaichang1, &QPushButton::clicked,
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
}

void DlgProductSetSmartCroppingOfBags::onUpdateCurrentPulse(double pulse)
{
	ui->btn_maichongxinhao1->setText(QString::number(pulse, 'f', 2));
}

void DlgProductSetSmartCroppingOfBags::pbtn_close_clicked()
{
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

void DlgProductSetSmartCroppingOfBags::btn_pingjunmaichong1_clicked()
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
		ui->btn_pingjunmaichong1->setText(value);
		globalStructSetConfig.pingjunmaichong1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_maichongxinhao1_clicked()
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
		ui->btn_maichongxinhao1->setText(value);
		globalStructSetConfig.maichongxinhao1 = value.toDouble();
	}
}

void DlgProductSetSmartCroppingOfBags::btn_hanggao1_clicked()
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
		ui->btn_hanggao1->setText(value);
		globalStructSetConfig.hanggao1 = value.toDouble();
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
		ui->btn_zuixiaodaichang1->setText(value);
		globalStructSetConfig.zuixiaodaichang1 = value.toDouble();
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
		ui->btn_zuidadaichang1->setText(value);
		globalStructSetConfig.zuidadaichang1 = value.toDouble();
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
		if (value.toDouble() < 0 || value.toDouble() > 300)
		{
			QMessageBox::warning(this, "提示", "请输入大于0小于300的数值");
			return;
		}
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
		ui->btn_xiangjizengyi1->setText(value);
		globalStructSetConfig.xiangjizengyi1 = value.toDouble();
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
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
		auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
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

void DlgProductSetSmartCroppingOfBags::ckb_xiangjizengyi_checked()
{
	auto& globalStructSetConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	globalStructSetConfig.isxiangjizengyi1 = ui->ckb_xiangjizengyi->isChecked();
}

