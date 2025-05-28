#pragma once

#include <QDialog>
#include "ui_DlgWarningIOSetConfig.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgWarningIOSetConfigClass; };
QT_END_NAMESPACE

class DlgWarningIOSetConfig : public QDialog
{
	Q_OBJECT

public:
	DlgWarningIOSetConfig(QWidget *parent = nullptr);
	~DlgWarningIOSetConfig();
private:
	void build_ui();
	void build_connect();

private:
	Ui::DlgWarningIOSetConfigClass *ui;
public slots:
	void pbtn_exit_clicked();
};
