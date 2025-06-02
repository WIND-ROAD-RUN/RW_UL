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

public:
	void read_config();
	void read_config_GeneralConfig();

private:
	Ui::ZipperScannerClass *ui;
};

