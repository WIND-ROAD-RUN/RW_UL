#include "DlgProductScore.h"

DlgProductScore::DlgProductScore(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgProductScoreClass())
{
	ui->setupUi(this);
}

DlgProductScore::~DlgProductScore()
{
	delete ui;
}

