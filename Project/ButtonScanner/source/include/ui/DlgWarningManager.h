#pragma once

#include <QDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class DlgWarningManagerClass; };
QT_END_NAMESPACE

class DlgWarningManager : public QDialog
{
	Q_OBJECT

public:
	DlgWarningManager(QWidget* parent = nullptr);
	~DlgWarningManager();
public:
	void build_connect();
	void build_ui();

private:
	Ui::DlgWarningManagerClass* ui;
public slots:
	void pbtn_exit_clicked();
	void cbox_cameraDisconnect1_clicked(bool checked);
	void cbox_cameraDisconnect2_clicked(bool checked);
	void cbox_cameraDisconnect3_clicked(bool checked);
	void cbox_cameraDisconnect4_clicked(bool checked);
	void cbox_workTrigger1_clicked(bool checked);
	void cbox_workTrigger2_clicked(bool checked);
	void cbox_workTrigger3_clicked(bool checked);
	void cbox_workTrigger4_clicked(bool checked);
	void cbox_airPressure_clicked(bool checked);
};
