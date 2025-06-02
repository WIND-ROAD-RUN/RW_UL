#include "ZipperScanner.h"

ZipperScanner::ZipperScanner(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::ZipperScannerClass())
{
	ui->setupUi(this);
}

ZipperScanner::~ZipperScanner()
{
	delete ui;
}

