#pragma once

#include <QDialog>
#include "ui_DlgProductSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductSetClass; };
QT_END_NAMESPACE

class DlgProductSetSmartCroppingOfBags : public QDialog
{
	Q_OBJECT

public:
	DlgProductSetSmartCroppingOfBags(QWidget *parent = nullptr);
	~DlgProductSetSmartCroppingOfBags();

public:
	void build_ui();
	void read_config();
	void build_connect();

private:
	Ui::DlgProductSetClass *ui;
};

