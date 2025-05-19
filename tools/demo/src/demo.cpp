#include "demo.h"

demo::demo(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::demoClass())
{
	ui->setupUi(this);
}

demo::~demo()
{
	delete ui;
}
