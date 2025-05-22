#include "DlgWarningManager.h"

DlgWarningManager::DlgWarningManager(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarningManagerClass())
{
	ui->setupUi(this);
	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
}

DlgWarningManager::~DlgWarningManager()
{
	delete ui;
}
