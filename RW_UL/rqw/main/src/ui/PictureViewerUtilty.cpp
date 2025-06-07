#include "PictureViewerUtilty.h"

#include "ui_PictureViewerUtilty.h"

PictureViewerUtilty::PictureViewerUtilty(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::PictureViewerUtiltyClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
}

PictureViewerUtilty::~PictureViewerUtilty()
{
	delete ui;
}

void PictureViewerUtilty::build_ui()
{
}

void PictureViewerUtilty::build_connect()
{
	connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &PictureViewerUtilty::pbtn_exit_clicked);
}

void PictureViewerUtilty::showEvent(QShowEvent* event)
{
	QMainWindow::showEvent(event);
	if (!path.isEmpty() && ui->label_imgDisplay) {
		QPixmap pixmap(path);
		if (!pixmap.isNull()) {
			ui->label_imgDisplay->setPixmap(pixmap.scaled(ui->label_imgDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		}
		else {
			ui->label_imgDisplay->clear();
			ui->label_imgDisplay->setText("ÎÞ·¨¼ÓÔØÍ¼Æ¬");
		}
	}
}

void PictureViewerUtilty::setImgPath(const QString& imgPath)
{
	path = imgPath;
}

void PictureViewerUtilty::pbtn_exit_clicked()
{
	this->close();
}
