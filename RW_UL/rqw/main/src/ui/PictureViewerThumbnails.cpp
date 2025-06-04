#include "PictureViewerThumbnails.h"

PictureViewerThumbnails::PictureViewerThumbnails(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::PictureViewerThumbnailsClass())
{
	ui->setupUi(this);
}

PictureViewerThumbnails::~PictureViewerThumbnails()
{
	delete ui;
}
