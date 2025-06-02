#pragma once

#include <QDialog>
#include "ui_DlgProductSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductSetClass; };
QT_END_NAMESPACE

class DlgProductSet : public QDialog
{
	Q_OBJECT

public:
	DlgProductSet(QWidget *parent = nullptr);
	~DlgProductSet();

private:
	Ui::DlgProductSetClass *ui;
};

