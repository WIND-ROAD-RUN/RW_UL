#include "AutomaticAnnotation.h"

AutomaticAnnotation::AutomaticAnnotation(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::AutomaticAnnotationClass())
{
	ui->setupUi(this);
}

AutomaticAnnotation::~AutomaticAnnotation()
{
	delete ui;
}
