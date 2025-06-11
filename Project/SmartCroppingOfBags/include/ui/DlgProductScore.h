#pragma once

#include <QDialog>
#include "ui_DlgProductScore.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductScoreClass; };
QT_END_NAMESPACE

class DlgProductScoreSmartCroppingOfBags : public QDialog
{
	Q_OBJECT

public:
	DlgProductScoreSmartCroppingOfBags(QWidget *parent = nullptr);
	~DlgProductScoreSmartCroppingOfBags();

public:
	void build_ui();
	void read_config();
	void build_connect();

private:
	Ui::DlgProductScoreClass *ui;
};

