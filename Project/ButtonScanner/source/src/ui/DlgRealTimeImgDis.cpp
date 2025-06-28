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
	connect(ui->pbtn_nextWork, &QPushButton::clicked,
		this, &DlgRealTimeImgDis::pbtn_nextWork_clicked);
	connect(ui->pbtn_preWork, &QPushButton::clicked,
		this, &DlgRealTimeImgDis::pbtn_preWork_clicked);
}

void DlgRealTimeImgDis::setMonitorValue(bool* isShow)
{
	if (!isShow)
	{
		return;
	}
	_isShow = isShow;
}

void DlgRealTimeImgDis::setMonitorDisImgIndex(int* index)
{
	if (!index)
	{
		return;
	}
	_currentDisImgIndex = index;
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

void DlgRealTimeImgDis::setShowImg(const QPixmap& image)
{
	ui->label_imgDis->setPixmap(image.scaled(ui->label_imgDis->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void DlgRealTimeImgDis::updateTitle(int index)
{
	switch (index)
	{
	case 0:
		setGboxTitle("1号工位");
		break;
	case 1:
		setGboxTitle("2号工位");
		break;
	case 2:
		setGboxTitle("3号工位");
		break;
	case 3:
		setGboxTitle("4号工位");
		break;
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

void DlgRealTimeImgDis::pbtn_nextWork_clicked()
{
	if (!_currentDisImgIndex)
	{
		return;
	}
	auto index = *_currentDisImgIndex;
	index += 1;
	index = (index + 4) % 4;
	updateTitle(index);
	*_currentDisImgIndex = index;
}

void DlgRealTimeImgDis::pbtn_preWork_clicked()
{
	if (!_currentDisImgIndex)
	{
		return;
	}
	auto index = *_currentDisImgIndex;
	index -= 1;
	index = (index + 4) % 4;
	updateTitle(index);
	*_currentDisImgIndex = index;
}
