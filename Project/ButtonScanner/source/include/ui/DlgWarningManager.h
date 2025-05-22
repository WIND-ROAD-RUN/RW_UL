#pragma once

#include <QDialog>
#include "ui_DlgWarningManager.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgWarningManagerClass; };
QT_END_NAMESPACE

class DlgWarningManager : public QDialog
{
	Q_OBJECT

public:
	DlgWarningManager(QWidget *parent = nullptr);
	~DlgWarningManager();

private:
	Ui::DlgWarningManagerClass *ui;
};
