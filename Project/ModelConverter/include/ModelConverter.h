#pragma once

#include <QMainWindow>
#include "ui_ModelConverter.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ModelConverterClass; };
QT_END_NAMESPACE

class ModelConverter : public QMainWindow
{
	Q_OBJECT

public:
	ModelConverter(QWidget *parent = nullptr);
	~ModelConverter();

private:
	Ui::ModelConverterClass *ui;
};
