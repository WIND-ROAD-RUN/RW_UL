#include "stdafx.h"
#include "DlgProductSet.h"
#include "NumberKeyboard.h"

#include "GlobalStruct.h"

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
	//隐藏拍照延时栏，该功能已弃用
	for (int i = 0; i < ui->hLayout_photography->count(); ++i) {
		QWidget* widget = ui->hLayout_photography->itemAt(i)->widget();
		if (widget) {
			widget->hide(); // 隐藏子控件
		}
	}

	_clickedLabel = new rw::rqw::ClickableLabel(this);
	_clickedLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	_dlgHideScoreSet = new DlgHideScoreSet(this);

	ui->horizontalLayout_firstRow->replaceWidget(ui->label_pic, _clickedLabel);
	ui->horizontalLayout_firstRow->update();
	delete ui->label_pic;
	readConfig();
	read_image();
	build_radioButton();
}

void DlgProductSet::readConfig()
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	//外径
	ui->rbtn_outsideDiameterEnable->setChecked(GlobalStructData.dlgProductSetConfig.outsideDiameterEnable);
	ui->pbtn_outsideDiameterValue->setText(QString::number(GlobalStructData.dlgProductSetConfig.outsideDiameterValue));
	ui->pbtn_outsideDiameterDeviation->setText(QString::number(GlobalStructData.dlgProductSetConfig.outsideDiameterDeviation));
	//图片
	ui->pbtn_photography->setText(QString::number(GlobalStructData.dlgProductSetConfig.photography));

	//吹气
	ui->pbtn_blowTime->setText(QString::number(GlobalStructData.dlgProductSetConfig.blowTime));

	//破边
	ui->rbtn_edgeDamageEnable->setChecked(GlobalStructData.dlgProductSetConfig.edgeDamageEnable);
	ui->pbtn_edgeDamageSimilarity->setText(QString::number(GlobalStructData.dlgProductSetConfig.edgeDamageSimilarity));
	ui->pbtn_edgeDamageArea->setText(QString::number(GlobalStructData.dlgProductSetConfig.edgeDamageArea));

	//崩口
	ui->rbtn_bengKou->setChecked(GlobalStructData.dlgProductSetConfig.bengKouEnabel);
	ui->pbtn_bengKouScore->setText(QString::number(GlobalStructData.dlgProductSetConfig.bengKouScore));

	//屏蔽范围
	ui->rbtn_shieldingRangeEnable->setChecked(GlobalStructData.dlgProductSetConfig.shieldingRangeEnable);
	ui->pbtn_outerRadius->setText(QString::number(GlobalStructData.dlgProductSetConfig.outerRadius));
	ui->pbtn_innerRadius->setText(QString::number(GlobalStructData.dlgProductSetConfig.innerRadius));

	//气孔
	ui->rbtn_poreEnable->setChecked(GlobalStructData.dlgProductSetConfig.poreEnable);
	ui->pbtn_poreEnableScore->setText(QString::number(GlobalStructData.dlgProductSetConfig.poreEnableScore));
	ui->pbtn_poreEnableArea->setText(QString::number(GlobalStructData.dlgProductSetConfig.poreEnableArea));

	//油漆
	ui->rbtn_paintEnable->setChecked(GlobalStructData.dlgProductSetConfig.paintEnable);
	ui->pbtn_paintEnableScore->setText(QString::number(GlobalStructData.dlgProductSetConfig.paintEnableScore));

	//孔数
	ui->rbtn_holesCountEnable->setChecked(GlobalStructData.dlgProductSetConfig.holesCountEnable);
	ui->ptn_holesCountValue->setText(QString::number(GlobalStructData.dlgProductSetConfig.holesCountValue));

	//孔心距
	ui->pbtn_holeCenterDistanceSimilarity->setText(QString::number(GlobalStructData.dlgProductSetConfig.holeCenterDistanceSimilarity));
	ui->pbtn_holeCenterDistanceValue->setText(QString::number(GlobalStructData.dlgProductSetConfig.holeCenterDistanceValue));

	//破眼
	ui->rbtn_brokenEyeEnable->setChecked(GlobalStructData.dlgProductSetConfig.brokenEyeEnable);
	ui->pbtn_brokenEyeSimilarity->setText(QString::number(GlobalStructData.dlgProductSetConfig.brokenEyeSimilarity));

	//裂痕
	ui->rbtn_crackEnable->setChecked(GlobalStructData.dlgProductSetConfig.crackEnable);
	ui->pbtn_crackSimilarity->setText(QString::number(GlobalStructData.dlgProductSetConfig.crackSimilarity));

	//孔径
	ui->rbtn_apertureEnable->setChecked(GlobalStructData.dlgProductSetConfig.apertureEnable);
	ui->pbtn_apertureValue->setText(QString::number(GlobalStructData.dlgProductSetConfig.apertureValue));
	ui->pbtn_apertureSimilarity->setText(QString::number(GlobalStructData.dlgProductSetConfig.apertureSimilarity));

	//指定色差
	ui->rbtn_specifyColorDifferenceEnable->setChecked(GlobalStructData.dlgProductSetConfig.specifyColorDifferenceEnable);
	ui->pbtn_specifyColorDifferenceR->setText(QString::number(GlobalStructData.dlgProductSetConfig.specifyColorDifferenceR));
	ui->pbtn_specifyColorDifferenceG->setText(QString::number(GlobalStructData.dlgProductSetConfig.specifyColorDifferenceG));
	ui->pbtn_specifyColorDifferenceB->setText(QString::number(GlobalStructData.dlgProductSetConfig.specifyColorDifferenceB));
	ui->pbtn_specifyColorDifferenceDeviation->setText(QString::number(GlobalStructData.dlgProductSetConfig.specifyColorDifferenceDeviation));

	//大色差
	ui->rbtn_largeColorDifferenceEnable->setChecked(GlobalStructData.dlgProductSetConfig.largeColorDifferenceEnable);
	ui->pbtn_largeColorDifferenceDeviation->setText(QString::number(GlobalStructData.dlgProductSetConfig.largeColorDifferenceDeviation));

	//磨石
	ui->rbtn_grindStoneEnable->setChecked(GlobalStructData.dlgProductSetConfig.grindStoneEnable);
	ui->pbtn_grindStoneScore->setText(QString::number(GlobalStructData.dlgProductSetConfig.grindStoneEnableScore));

	//破眼
	ui->rbtn_blockEyeEnable->setChecked(GlobalStructData.dlgProductSetConfig.blockEyeEnable);
	ui->pbtn_blockEyeScore->setText(QString::number(GlobalStructData.dlgProductSetConfig.blockEyeEnableScore));

	//料头
	ui->rbtn_materialHeadEnable->setChecked(GlobalStructData.dlgProductSetConfig.materialHeadEnable);
	ui->pbtn_materialHeadScore->setText(QString::number(GlobalStructData.dlgProductSetConfig.materialHeadEnableScore));

	//吹气
	ui->pbtn_blowTime->setText(QString::number(get_blowTime()));

	//小气孔
	ui->pbtn_smallPoreEnableArea->setText(QString::number(GlobalStructData.dlgProductSetConfig.smallPoreEnableArea));
	ui->pbtn_smallPoreEnableScore->setText(QString::number(GlobalStructData.dlgProductSetConfig.smallPoreEnableScore));
	ui->rbtn_smallPoreEnable->setChecked(GlobalStructData.dlgProductSetConfig.smallPoreEnable);

}

float DlgProductSet::get_blowTime()
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	auto outsideDiameterValue = GlobalStructData.dlgProductSetConfig.outsideDiameterValue;
	auto beltSpeed = GlobalStructData.dlgProduceLineSetConfig.motorSpeed;
	auto blowTime = outsideDiameterValue / beltSpeed * 1000 / 2;
	return blowTime;
}

void DlgProductSet::read_image()
{
	QString imagePath = ":/ButtonScanner/image/product.png";
	QPixmap pixmap(imagePath);

	if (pixmap.isNull()) {
		QMessageBox::critical(this, "Error", "无法加载图片。");
		return;
	}
	_clickedLabel->setPixmap(pixmap);
}

void DlgProductSet::build_connect()
{
	QObject::connect(ui->pbtn_close, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_close_clicked);

	QObject::connect(ui->pbtn_photography, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_photography_clicked);

	QObject::connect(ui->pbtn_blowTime, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_blowTime_clicked);

	QObject::connect(_clickedLabel, &rw::rqw::ClickableLabel::clicked,
		this, &DlgProductSet::clickedLabel_clicked);

	//外径
	QObject::connect(ui->pbtn_outsideDiameterValue, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_outsideDiameterValue_clicked);
	QObject::connect(ui->pbtn_outsideDiameterDeviation, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_outsideDiameterDeviation_clicked);
	QObject::connect(ui->rbtn_outsideDiameterEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_outsideDiameterEnable_checked);

	//屏蔽范围
	QObject::connect(ui->rbtn_shieldingRangeEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_shieldingRangeEnable_checked);
	QObject::connect(ui->pbtn_outerRadius, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_outerRadius_clicked);
	QObject::connect(ui->pbtn_innerRadius, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_innerRadius_clicked);

	//孔数
	QObject::connect(ui->ptn_holesCountValue, &QPushButton::clicked,
		this, &DlgProductSet::ptn_holesCountValue_clicked);
	QObject::connect(ui->rbtn_holesCountEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_holesCountEnable_checked);

	//孔心距
	QObject::connect(ui->pbtn_holeCenterDistanceValue, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_holeCenterDistanceValue_clicked);
	QObject::connect(ui->pbtn_holeCenterDistanceSimilarity, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_holeCenterDistanceSimilarity_clicked);
	QObject::connect(ui->rbtn_holeCenterDistanceEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_holeCenterDistanceEnable_checked);

	//破眼
	QObject::connect(ui->pbtn_brokenEyeSimilarity, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_brokenEyeSimilarity_clicked);
	QObject::connect(ui->rbtn_brokenEyeEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_brokenEyeEnable_checked);

	//崩口
	QObject::connect(ui->pbtn_bengKouScore, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_bengKouScore_clicked);
	QObject::connect(ui->rbtn_bengKou, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_bengKou_checked);

	//裂痕
	QObject::connect(ui->pbtn_crackSimilarity, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_crackSimilarity_clicked);
	QObject::connect(ui->rbtn_crackEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_crackEnable_checked);

	//孔径
	QObject::connect(ui->pbtn_apertureValue, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_apertureValue_clicked);
	QObject::connect(ui->pbtn_apertureSimilarity, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_apertureSimilarity_clicked);
	QObject::connect(ui->rbtn_apertureEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_apertureEnable_checked);

	//指定色差
	QObject::connect(ui->pbtn_specifyColorDifferenceR, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_specifyColorDifferenceR_clicked);
	QObject::connect(ui->pbtn_specifyColorDifferenceG, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_specifyColorDifferenceG_clicked);
	QObject::connect(ui->pbtn_specifyColorDifferenceB, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_specifyColorDifferenceB_clicked);
	QObject::connect(ui->pbtn_specifyColorDifferenceDeviation, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_specifyColorDifferenceDeviation_clicked);
	QObject::connect(ui->rbtn_specifyColorDifferenceEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_specifyColorDifferenceEnable_checked);

	//大色差
	QObject::connect(ui->pbtn_largeColorDifferenceDeviation, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_largeColorDifferenceDeviation_clicked);
	QObject::connect(ui->rbtn_largeColorDifferenceEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_largeColorDifferenceEnable_checked);

	//破边
	QObject::connect(ui->pbtn_edgeDamageSimilarity, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_edgeDamageSimilarity_clicked);
	QObject::connect(ui->pbtn_edgeDamageArea, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_edgeDamageArea_clicked);
	QObject::connect(ui->rbtn_edgeDamageEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_edgeDamageEnable_checked);

	//气孔
	QObject::connect(ui->pbtn_poreEnableScore, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_poreEnableScore_clicked);
	QObject::connect(ui->rbtn_poreEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_poreEnable_checked);
	QObject::connect(ui->pbtn_poreEnableArea, &QRadioButton::clicked,
		this, &DlgProductSet::pbtn_poreEnableArea_clicked);

	//小气孔
	QObject::connect(ui->pbtn_smallPoreEnableArea, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_smallPoreEnableArea_clicked);
	QObject::connect(ui->pbtn_smallPoreEnableScore, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_smallPoreEnableScore_clicked);
	QObject::connect(ui->rbtn_smallPoreEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_smallPoreEnable_checked);

	//油漆
	QObject::connect(ui->pbtn_paintEnableScore, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_paintEnableScore_clicked);
	QObject::connect(ui->rbtn_paintEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_paintEnable_checked);

	//磨石
	QObject::connect(ui->pbtn_grindStoneScore, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_grindStoneScore_clicked);
	QObject::connect(ui->rbtn_grindStoneEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_grindStoneEnable_checked);

	//堵眼
	QObject::connect(ui->pbtn_blockEyeScore, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_blockEyeScore_clicked);
	QObject::connect(ui->rbtn_blockEyeEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_blockEyeEnable_checked);

	//料头
	QObject::connect(ui->pbtn_materialHeadScore, &QPushButton::clicked,
		this, &DlgProductSet::pbtn_materialHeadScore_clicked);
	QObject::connect(ui->rbtn_materialHeadEnable, &QRadioButton::clicked,
		this, &DlgProductSet::rbtn_materialHeadEnable_checked);


}

void DlgProductSet::build_radioButton()
{
	ui->rbtn_poreEnable->setAutoExclusive(false);
	ui->rbtn_paintEnable->setAutoExclusive(false);
	ui->rbtn_grindStoneEnable->setAutoExclusive(false);
	ui->rbtn_blockEyeEnable->setAutoExclusive(false);
	ui->rbtn_shieldingRangeEnable->setAutoExclusive(false);
}

void DlgProductSet::pbtn_outsideDiameterValue_clicked() {
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_outsideDiameterValue->setText(value);
		GlobalStructData.dlgProductSetConfig.outsideDiameterValue = value.toDouble();
		// 计算并更新 blowTime
		ui->pbtn_blowTime->setText(QString::number(get_blowTime()));
		GlobalStructData.dlgProductSetConfig.blowTime = ui->pbtn_blowTime->text().toDouble();
	}
}

void DlgProductSet::pbtn_outsideDiameterDeviation_clicked()
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_outsideDiameterDeviation->setText(value);
		GlobalStructData.dlgProductSetConfig.outsideDiameterDeviation = value.toDouble();
	}
}

void DlgProductSet::pbtn_photography_clicked()
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_photography->setText(value);
		GlobalStructData.dlgProductSetConfig.photography = value.toDouble();
	}
}

void DlgProductSet::pbtn_blowTime_clicked()
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blowTime->setText(value);
		GlobalStructData.dlgProductSetConfig.blowTime = value.toDouble();
	}
}

void DlgProductSet::pbtn_outerRadius_clicked()
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_outerRadius->setText(value);
		GlobalStructData.dlgProductSetConfig.outerRadius = value.toDouble();
	}
}

void DlgProductSet::pbtn_innerRadius_clicked()
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_innerRadius->setText(value);
		GlobalStructData.dlgProductSetConfig.innerRadius = value.toDouble();
	}
}

void DlgProductSet::ptn_holesCountValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 10)
		{
			QMessageBox::warning(this, "提示", "请输入0-10之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->ptn_holesCountValue->setText(value);
		GlobalStructData.dlgProductSetConfig.holesCountValue = value.toDouble();
	}
}

void DlgProductSet::pbtn_brokenEyeSimilarity_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_brokenEyeSimilarity->setText(value);
		GlobalStructData.dlgProductSetConfig.brokenEyeSimilarity = value.toDouble();
	}
}

void DlgProductSet::pbtn_crackSimilarity_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_crackSimilarity->setText(value);
		GlobalStructData.dlgProductSetConfig.crackSimilarity = value.toDouble();
	}
}

void DlgProductSet::pbtn_apertureValue_clicked()
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_apertureValue->setText(value);
		GlobalStructData.dlgProductSetConfig.apertureValue = value.toDouble();
	}
}

void DlgProductSet::pbtn_apertureSimilarity_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_apertureSimilarity->setText(value);
		GlobalStructData.dlgProductSetConfig.apertureSimilarity = value.toDouble();
	}
}

void DlgProductSet::pbtn_holeCenterDistanceValue_clicked()
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
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_holeCenterDistanceValue->setText(value);
		GlobalStructData.dlgProductSetConfig.holeCenterDistanceValue = value.toDouble();
	}
}

void DlgProductSet::pbtn_holeCenterDistanceSimilarity_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_holeCenterDistanceSimilarity->setText(value);
		GlobalStructData.dlgProductSetConfig.holeCenterDistanceSimilarity = value.toDouble();
	}
}

void DlgProductSet::pbtn_specifyColorDifferenceR_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 255)
		{
			QMessageBox::warning(this, "提示", "请输入0-255之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_specifyColorDifferenceR->setText(value);
		GlobalStructData.dlgProductSetConfig.specifyColorDifferenceR = value.toDouble();
	}
}

void DlgProductSet::pbtn_specifyColorDifferenceG_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 255)
		{
			QMessageBox::warning(this, "提示", "请输入0-255之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_specifyColorDifferenceG->setText(value);
		GlobalStructData.dlgProductSetConfig.specifyColorDifferenceG = value.toDouble();
	}
}

void DlgProductSet::pbtn_specifyColorDifferenceB_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 255)
		{
			QMessageBox::warning(this, "提示", "请输入0-255之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_specifyColorDifferenceB->setText(value);
		GlobalStructData.dlgProductSetConfig.specifyColorDifferenceB = value.toDouble();
	}
}

void DlgProductSet::pbtn_specifyColorDifferenceDeviation_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 255)
		{
			QMessageBox::warning(this, "提示", "请输入0-255之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_specifyColorDifferenceDeviation->setText(value);
		GlobalStructData.dlgProductSetConfig.specifyColorDifferenceDeviation = value.toDouble();
	}
}

void DlgProductSet::pbtn_largeColorDifferenceDeviation_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 255)
		{
			QMessageBox::warning(this, "提示", "请输入0-255之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_largeColorDifferenceDeviation->setText(value);
		GlobalStructData.dlgProductSetConfig.largeColorDifferenceDeviation = value.toDouble();
	}
}

void DlgProductSet::pbtn_edgeDamageSimilarity_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_edgeDamageSimilarity->setText(value);
		GlobalStructData.dlgProductSetConfig.edgeDamageSimilarity = value.toDouble();
	}
}

void DlgProductSet::pbtn_edgeDamageArea_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 )
		{
			QMessageBox::warning(this, "提示", "请输入大于0的数值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_edgeDamageArea->setText(value);
		GlobalStructData.dlgProductSetConfig.edgeDamageArea = value.toDouble();
	}
}

void DlgProductSet::rbtn_bengKou_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.bengKouEnabel = checked;
}

void DlgProductSet::pbtn_bengKouScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_bengKouScore->setText(value);
		GlobalStructData.dlgProductSetConfig.bengKouScore = value.toDouble();
	}
}

void DlgProductSet::pbtn_poreEnableScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_poreEnableScore->setText(value);
		GlobalStructData.dlgProductSetConfig.poreEnableScore = value.toDouble();
	}
}

void DlgProductSet::pbtn_smallPoreEnableScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_smallPoreEnableScore->setText(value);
		GlobalStructData.dlgProductSetConfig.smallPoreEnableScore = value.toDouble();
	}
}

void DlgProductSet::pbtn_smallPoreEnableArea_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 200000)
		{
			QMessageBox::warning(this, "提示", "请输入0-200000之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_smallPoreEnableArea->setText(value);
		GlobalStructData.dlgProductSetConfig.smallPoreEnableArea = value.toDouble();
	}
}

void DlgProductSet::pbtn_paintEnableScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_paintEnableScore->setText(value);
		GlobalStructData.dlgProductSetConfig.paintEnableScore = value.toDouble();
	}
}

void DlgProductSet::pbtn_grindStoneScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_grindStoneScore->setText(value);
		GlobalStructData.dlgProductSetConfig.grindStoneEnableScore = value.toDouble();
	}
}

void DlgProductSet::pbtn_blockEyeScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_blockEyeScore->setText(value);
		GlobalStructData.dlgProductSetConfig.blockEyeEnableScore = value.toDouble();
	}
}

void DlgProductSet::pbtn_materialHeadScore_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 100)
		{
			QMessageBox::warning(this, "提示", "请输入0-100之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_materialHeadScore->setText(value);
		GlobalStructData.dlgProductSetConfig.materialHeadEnableScore = value.toDouble();
	}
}

void DlgProductSet::pbtn_poreEnableArea_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		if (value.toDouble() < 0 || value.toDouble() > 200000)
		{
			QMessageBox::warning(this, "提示", "请输入0-200000之间的值");
			return;
		}
		auto& GlobalStructData = GlobalStructData::getInstance();
		ui->pbtn_poreEnableArea->setText(value);
		GlobalStructData.dlgProductSetConfig.poreEnableArea = value.toDouble();
	}
}

void DlgProductSet::rbtn_outsideDiameterEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.outsideDiameterEnable = checked;
}

void DlgProductSet::rbtn_edgeDamageEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.edgeDamageEnable = checked;
}

void DlgProductSet::rbtn_shieldingRangeEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.shieldingRangeEnable = checked;
}

void DlgProductSet::rbtn_poreEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.poreEnable = checked;
}

void DlgProductSet::rbtn_smallPoreEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.smallPoreEnable = checked;
}

void DlgProductSet::rbtn_paintEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.paintEnable = checked;
}

void DlgProductSet::rbtn_holesCountEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.holesCountEnable = checked;
}

void DlgProductSet::rbtn_brokenEyeEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.brokenEyeEnable = checked;
}

void DlgProductSet::rbtn_crackEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.crackEnable = checked;
}

void DlgProductSet::rbtn_apertureEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.apertureEnable = checked;
}

void DlgProductSet::rbtn_holeCenterDistanceEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.holeCenterDistanceEnable = checked;
}

void DlgProductSet::rbtn_specifyColorDifferenceEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.specifyColorDifferenceEnable = checked;
}

void DlgProductSet::rbtn_largeColorDifferenceEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.largeColorDifferenceEnable = checked;
}

void DlgProductSet::rbtn_grindStoneEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.grindStoneEnable = checked;
}

void DlgProductSet::rbtn_blockEyeEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.blockEyeEnable = checked;
}

void DlgProductSet::rbtn_materialHeadEnable_checked(bool checked)
{
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.dlgProductSetConfig.materialHeadEnable = checked;
}

void DlgProductSet::pbtn_close_clicked() {
	auto& GlobalStructData = GlobalStructData::getInstance();
	GlobalStructData.saveDlgProductSetConfig();
	this->close();
}

void DlgProductSet::clickedLabel_clicked()
{
	_dlgHideScoreSet->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	_dlgHideScoreSet->show();
}