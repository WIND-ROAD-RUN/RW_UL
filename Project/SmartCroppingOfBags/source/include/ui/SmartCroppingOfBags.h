#pragma once

#include <QMainWindow>
#include "ui_SmartCroppingOfBags.h"

QT_BEGIN_NAMESPACE
namespace Ui { class SmartCroppingOfBagsClass; };
QT_END_NAMESPACE

class SmartCroppingOfBags : public QMainWindow
{
	Q_OBJECT

public:
	SmartCroppingOfBags(QWidget *parent = nullptr);
	~SmartCroppingOfBags();

private:
	Ui::SmartCroppingOfBagsClass *ui;
};

