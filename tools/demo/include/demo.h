#pragma once

#include <QMainWindow>
#include "ui_demo.h"

QT_BEGIN_NAMESPACE
namespace Ui { class demoClass; };
QT_END_NAMESPACE

class demo : public QMainWindow
{
	Q_OBJECT

public:
	demo(QWidget *parent = nullptr);
	~demo();

private:
	Ui::demoClass *ui;
};
