#include "DlgWarningManager.h"

DlgWarningManager::DlgWarningManager(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarningManagerClass())
{
	ui->setupUi(this);
}

DlgWarningManager::~DlgWarningManager()
{
	delete ui;
}
