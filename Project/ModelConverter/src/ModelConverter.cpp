#include "ModelConverter.h"

ModelConverter::ModelConverter(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::ModelConverterClass())
{
	ui->setupUi(this);
}

ModelConverter::~ModelConverter()
{
	delete ui;
}
