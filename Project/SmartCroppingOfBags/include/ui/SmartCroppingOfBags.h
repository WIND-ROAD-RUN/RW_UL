#pragma once

#include <QMainWindow>
#include "ui_SmartCroppingOfBags.h"
#include "DlgProductSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class SmartCroppingOfBagsClass; };
QT_END_NAMESPACE

class SmartCroppingOfBags : public QMainWindow
{
	Q_OBJECT

public:
	SmartCroppingOfBags(QWidget *parent = nullptr);
	~SmartCroppingOfBags();

public:
	DlgProductSet* _dlgProductSet = nullptr;

public:
	void build_ui();
	void build_connect();
	void build_camera();

	void build_SmartCroppingOfBagsData();
	void build_DlgProductSetData();
	void build_DlgProductScore();

	void build_imageProcessorModule();
	void build_imageSaveEngine();

public:
	void read_config();
	void read_config_GeneralConfig();
	void read_config_ScoreConfig();
	void read_config_SetConfig();

private:
	Ui::SmartCroppingOfBagsClass *ui;
};

