#include "DlgRealTimeImgDis.h"

#include "ui_DlgRealTimeImgDis.h"

DlgRealTimeImgDis::DlgRealTimeImgDis(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgRealTimeImgDisClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
}

DlgRealTimeImgDis::~DlgRealTimeImgDis()
{
	delete ui;
}

void DlgRealTimeImgDis::build_ui()
{
	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
}

void DlgRealTimeImgDis::build_connect()
{
	connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &DlgRealTimeImgDis::pbtn_exit_clicked);
}

void DlgRealTimeImgDis::setMonitorValue(bool* isShow)
{
	if (!isShow)
	{
		return;
	}
	_isShow = isShow;
}

void DlgRealTimeImgDis::setGboxTitle(const QString& title)
{
	ui->gBox_imgDis->setTitle(title);
}

void DlgRealTimeImgDis::showEvent(QShowEvent* event)
{
	if (_isShow)
	{
		*_isShow = true;
	}
}

void DlgRealTimeImgDis::pbtn_exit_clicked()
{
	if (_isShow)
	{
		*_isShow = false;
	}
	this->close();
}
