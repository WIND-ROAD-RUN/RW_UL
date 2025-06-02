#include "DlgWarningIOSetConfig.h"

#include"GlobalStruct.h"
#include"NumberKeyboard.h"
#include <QMessageBox>


std::vector<std::vector<int>> DlgWarningIOSetConfig::DIFindAllDuplicateIndices()
{
	auto& config = GlobalStructData::getInstance().warningIOSetConfig;
	std::vector<int> values = {
		config.DIStart,
		config.DIStop,
		config.DIShutdownComputer,
		config.DIAirPressure,
		config.DICameraTrigger1,
		config.DICameraTrigger2,
		config.DICameraTrigger3,
		config.DICameraTrigger4
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

void DlgWarningIOSetConfig::setDIErrorInfo(const std::vector<std::vector<int>>& index)
{
	ui->label_startWarn->clear();
	ui->label_stopWarn->clear();
	ui->label_shutdownComputerWarn->clear();
	ui->label_airPressureWarn->clear();
	ui->label_cameraTriggerWarn1->clear();
	ui->label_cameraTriggerWarn2->clear();
	ui->label_cameraTriggerWarn3->clear();
	ui->label_cameraTriggerWarn4->clear();
	for (const auto & classic:index)
	{
		for (const auto& item : classic)
		{
			setDIErrorInfo(item);
		}
	}
}

void DlgWarningIOSetConfig::setDIErrorInfo(int index)
{
	QString text = "重复数值";
	switch (index)
	{
	case 0:
		ui->label_startWarn->setText(text);
		break;
	case 1:
		ui->label_stopWarn->setText(text);
		break;
	case 2:
		ui->label_shutdownComputerWarn->setText(text);
		break;
	case 3:
		ui->label_airPressureWarn->setText(text);
		break;
	case 4:
		ui->label_cameraTriggerWarn1->setText(text);
		break;
	case 5:
		ui->label_cameraTriggerWarn2->setText(text);
		break;
	case 6:
		ui->label_cameraTriggerWarn3->setText(text);
		break;
	case 7:
		ui->label_cameraTriggerWarn4->setText(text);
		break;
	}
}

std::vector<std::vector<int>> DlgWarningIOSetConfig::DOFindAllDuplicateIndices()
{
	auto& config = GlobalStructData::getInstance().warningIOSetConfig;
	std::vector<int> values = {
		config.DOMotoPower,
		config.DOBlow1,
		config.DOBlow2,
		config.DOBlow3,
		config.DOBlow4,
		config.DOGreenLight,
		config.DORedLight,
		config.DOUpLight,
		config.DOSideLight,
		config.DODownLight,
		config.DOStrobeLight,
		config.DOStartBelt
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

void DlgWarningIOSetConfig::setDOErrorInfo(const std::vector<std::vector<int>>& index)
{
	ui->label_motoPowerWarn->clear();
	ui->label_blowWarn1->clear();
	ui->label_blowWarn2->clear();
	ui->label_blowWarn3->clear();
	ui->label_blowWarn4->clear();
	ui->label_greenLight->clear();
	ui->label_redLightWarn->clear();
	ui->label_upLightWarn->clear();
	ui->label_sideLightWarn->clear();
	ui->lavbel_downWarn->clear();
	ui->label_storeLightWarn->clear();
	ui->label_startBelt->clear();
	for (const auto& classic : index)
	{
		for (const auto& item : classic)
		{
			setDOErrorInfo(item);
		}
	}
}

void DlgWarningIOSetConfig::setDOErrorInfo(int index)
{
	QString text = "重复数值";
	switch (index)
	{
	case 0:
		ui->label_motoPowerWarn->setText(text);
		break;
	case 1:
		ui->label_blowWarn1->setText(text);
		break;
	case 2:
		ui->label_blowWarn2->setText(text);
		break;
	case 3:
		ui->label_blowWarn3->setText(text);
		break;
	case 4:
		ui->label_blowWarn4->setText(text);
		break;
	case 5:
		ui->label_greenLight->setText(text);
		break;
	case 6:
		ui->label_redLightWarn->setText(text);
		break;
	case 7:
		ui->label_upLightWarn->setText(text);
		break;
	case 8:
		ui->label_sideLightWarn->setText(text);
		break;
	case 9:
		ui->lavbel_downWarn->setText(text);
		break;
	case 10:
		ui->label_storeLightWarn->setText(text);
		break;
	case 11:
		ui->label_startBelt->setText(text);
		break;
	}
}

DlgWarningIOSetConfig::DlgWarningIOSetConfig(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarningIOSetConfigClass())
{
	ui->setupUi(this);

	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	read_config();
	build_ui();
	build_connect();
}

DlgWarningIOSetConfig::~DlgWarningIOSetConfig()
{
	delete ui;
}

void DlgWarningIOSetConfig::build_ui()
{
	auto DIIndices=DIFindAllDuplicateIndices();
	setDIErrorInfo(DIIndices);
	auto indicesDO = DOFindAllDuplicateIndices();
	setDOErrorInfo(indicesDO);
}

void DlgWarningIOSetConfig::build_connect()
{
	connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_exit_clicked);

	//DI
	connect(ui->pbtn_DIStartValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIStartValue_clicked);
	connect(ui->pbtn_DIStopValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIStopValue_clicked);
	connect(ui->pbtn_DIShutdownComputerValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIShutdownComputerValue_clicked);
	connect(ui->pbtn_DIAirPressureValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DIAirPressureValue_clicked);
	connect(ui->pbtn_DICameraTriggerValue1, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue1_clicked);
	connect(ui->pbtn_DICameraTriggerValue2, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue2_clicked);
	connect(ui->pbtn_DICameraTriggerValue3, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue3_clicked);
	connect(ui->pbtn_DICameraTriggerValue4, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DICameraTriggerValue4_clicked);

	//DO
	connect(ui->pbtn_DOMotoPowerValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOMotoPowerValue_clicked);
	connect(ui->pbtn_DOBlowTime1Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime1Value_clicked);
	connect(ui->pbtn_DOBlowTime2Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime2Value_clicked);
	connect(ui->pbtn_DOBlowTime3Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime3Value_clicked);
	connect(ui->pbtn_DOBlowTime4Value, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOBlowTime4Value_clicked);
	connect(ui->pbtn_DOGreenLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOGreenLightValue_clicked);
	connect(ui->pbtn_DORedLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DORedLightValue_clicked);
	connect(ui->pbtn_DOUpLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOUpLightValue_clicked);
	connect(ui->pbtn_DOSideLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOSideLightValue_clicked);
	connect(ui->pbtn_DODownLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DODownLightValue_clicked);
	connect(ui->pbtn_DOStoreLightValue, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOStoreLightValue_clicked);
	connect(ui->pbtn_DOStartBelt, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_DOStartBelt_clicked);
}

void DlgWarningIOSetConfig::read_config()
{
	auto& config = GlobalStructData::getInstance().warningIOSetConfig;
	ui->pbtn_DIAirPressureValue->setText(QString::number(config.DIAirPressure));
	ui->pbtn_DICameraTriggerValue1->setText(QString::number(config.DICameraTrigger1));
	ui->pbtn_DICameraTriggerValue2->setText(QString::number(config.DICameraTrigger2));
	ui->pbtn_DICameraTriggerValue3->setText(QString::number(config.DICameraTrigger3));
	ui->pbtn_DICameraTriggerValue4->setText(QString::number(config.DICameraTrigger4));
	ui->pbtn_DIShutdownComputerValue->setText(QString::number(config.DIShutdownComputer));
	ui->pbtn_DIStartValue->setText(QString::number(config.DIStart));
	ui->pbtn_DIStopValue->setText(QString::number(config.DIStop));
	ui->pbtn_DOBlowTime1Value->setText(QString::number(config.DOBlow1));
	ui->pbtn_DOBlowTime2Value->setText(QString::number(config.DOBlow2));
	ui->pbtn_DOBlowTime3Value->setText(QString::number(config.DOBlow3));
	ui->pbtn_DOBlowTime4Value->setText(QString::number(config.DOBlow4));
	ui->pbtn_DOMotoPowerValue->setText(QString::number(config.DOMotoPower));
	ui->pbtn_DOGreenLightValue->setText(QString::number(config.DOGreenLight));
	ui->pbtn_DORedLightValue->setText(QString::number(config.DORedLight));
	ui->pbtn_DOSideLightValue->setText(QString::number(config.DOSideLight));
	ui->pbtn_DOUpLightValue->setText(QString::number(config.DOUpLight));
	ui->pbtn_DODownLightValue->setText(QString::number(config.DODownLight));
	ui->pbtn_DOStoreLightValue->setText(QString::number(config.DOStrobeLight));
	ui->pbtn_DOStartBelt->setText(QString::number(config.DOStartBelt));
}

void DlgWarningIOSetConfig::pbtn_exit_clicked()
{
	this->close();
}

void DlgWarningIOSetConfig::pbtn_DIStartValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue<0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DIStartValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DIStart = numValue;
		auto index=DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DIStopValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DIStopValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DIStop = numValue;
		auto index = DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DIShutdownComputerValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DIShutdownComputerValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DIShutdownComputer = numValue;
		auto index = DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DIAirPressureValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DIAirPressureValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DIAirPressure = numValue;
		auto index = DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue1_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DICameraTriggerValue1->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DICameraTrigger1 = numValue;
		auto index = DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue2_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DICameraTriggerValue2->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DICameraTrigger2 = numValue;
		auto index = DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue3_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DICameraTriggerValue3->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DICameraTrigger3 = numValue;
		auto index = DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DICameraTriggerValue4_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DICameraTriggerValue4->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DICameraTrigger4 = numValue;
		auto index = DIFindAllDuplicateIndices();
		setDIErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime1Value_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOBlowTime1Value->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOBlow1 = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime2Value_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOBlowTime2Value->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOBlow2 = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime3Value_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOBlowTime3Value->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOBlow3 = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOBlowTime4Value_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOBlowTime4Value->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOBlow4 = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOGreenLightValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOGreenLightValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOGreenLight = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DORedLightValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DORedLightValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DORedLight = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOUpLightValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOUpLightValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOUpLight = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOSideLightValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOSideLightValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOSideLight = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DODownLightValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DODownLightValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DODownLight = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOStoreLightValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOStoreLightValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOStrobeLight = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOStartBelt_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOStartBelt->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOStartBelt = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}

void DlgWarningIOSetConfig::pbtn_DOMotoPowerValue_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		auto numValue = value.toInt();
		if (numValue < 0)
		{
			QMessageBox::warning(this, QString("提示"), QString("请输入大于0的数值"));
			return;
		}

		ui->pbtn_DOMotoPowerValue->setText(value);
		GlobalStructData::getInstance().warningIOSetConfig.DOMotoPower = numValue;
		auto index = DOFindAllDuplicateIndices();
		setDOErrorInfo(index);
	}
}
