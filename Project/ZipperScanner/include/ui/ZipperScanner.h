#pragma once

#include <QMainWindow>
#include "ui_ZipperScanner.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ZipperScannerClass; };
QT_END_NAMESPACE

class ZipperScanner : public QMainWindow
{
	Q_OBJECT

public:
	ZipperScanner(QWidget *parent = nullptr);
	~ZipperScanner();
public :
	void build_ui();
	void build_connect();

	void build_ZipperScannerData();
public:
	void read_config();
	void read_config_GeneralConfig();
	void read_config_ScoreConfig();
	void read_config_SetConfig();

private:
	Ui::ZipperScannerClass *ui;
};

