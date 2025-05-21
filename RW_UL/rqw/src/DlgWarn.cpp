#include "DlgWarn.h"

#include "ui_DlgWarn.h"

DlgWarn::DlgWarn(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarnClass())
{
	ui->setupUi(this);
	build_ui();
	build_connect();
}

DlgWarn::~DlgWarn()
{
	delete ui;
}

void DlgWarn::build_ui()
{
	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
}

void DlgWarn::build_connect()
{
	connect(ui->pbtn_accept,&QPushButton::clicked,
		this,&DlgWarn::pbtn_accept_clicked);
	connect(ui->pbtn_ignore, &QPushButton::clicked,
		this, &DlgWarn::pbtn_ignore_clicked);
}

void DlgWarn::pbtn_ignore_clicked()
{
	this->hide();
}

void DlgWarn::pbtn_accept_clicked()
{
	this->hide();
}
