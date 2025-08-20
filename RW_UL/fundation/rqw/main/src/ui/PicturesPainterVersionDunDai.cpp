#include "PicturesPainterVersionDunDai.h"

PicturesPainterVersionDunDai::PicturesPainterVersionDunDai(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::PicturesPainterVersionDunDaiClass())
{
	ui->setupUi(this);
}

PicturesPainterVersionDunDai::~PicturesPainterVersionDunDai()
{
	delete ui;
}

