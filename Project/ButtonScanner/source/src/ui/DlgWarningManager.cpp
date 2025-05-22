#include "DlgWarningManager.h"

DlgWarningManager::DlgWarningManager(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::DlgWarningManagerClass())
{
	ui->setupUi(this);
	this->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	build_ui();
	build_connect();
}

DlgWarningManager::~DlgWarningManager()
{
	delete ui;
}

void DlgWarningManager::build_connect()
{
	connect(ui->pbtn_exit, &QPushButton::clicked, this, &DlgWarningManager::pbtn_exit_clicked);
}

void DlgWarningManager::build_ui()
{
}

void DlgWarningManager::pbtn_exit_clicked()
{
	this->close();
}

