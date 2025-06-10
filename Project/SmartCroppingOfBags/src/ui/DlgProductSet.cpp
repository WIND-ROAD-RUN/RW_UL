#include "DlgProductSet.h"

DlgProductSet::DlgProductSet(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductSetClass())
{
	ui->setupUi(this);
}

DlgProductSet::~DlgProductSet()
{
	delete ui;
}

