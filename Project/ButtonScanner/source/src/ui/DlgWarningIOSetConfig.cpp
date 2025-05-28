#include "DlgWarningIOSetConfig.h"

DlgWarningIOSetConfig::DlgWarningIOSetConfig(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarningIOSetConfigClass())
{
	ui->setupUi(this);

	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	build_ui();
	build_connect();
}

DlgWarningIOSetConfig::~DlgWarningIOSetConfig()
{
	delete ui;
}

void DlgWarningIOSetConfig::build_ui()
{

}

void DlgWarningIOSetConfig::build_connect()
{
	connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &DlgWarningIOSetConfig::pbtn_exit_clicked);
}

void DlgWarningIOSetConfig::pbtn_exit_clicked()
{
	this->close();
}
