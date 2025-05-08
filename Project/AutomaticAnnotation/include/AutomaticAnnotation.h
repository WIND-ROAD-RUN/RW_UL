#pragma once

#include <QMainWindow>
#include "ui_AutomaticAnnotation.h"

QT_BEGIN_NAMESPACE
namespace Ui { class AutomaticAnnotationClass; };
QT_END_NAMESPACE

class AutomaticAnnotation : public QMainWindow
{
	Q_OBJECT

public:
	AutomaticAnnotation(QWidget *parent = nullptr);
	~AutomaticAnnotation();

private:
	Ui::AutomaticAnnotationClass *ui;
};
