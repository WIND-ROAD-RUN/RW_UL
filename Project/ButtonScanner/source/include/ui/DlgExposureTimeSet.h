#pragma once

#include <QDialog>
#include "ui_DlgExposureTimeSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgExposureTimeSetClass; };
QT_END_NAMESPACE

class DlgExposureTimeSet : public QDialog
{
	Q_OBJECT

public:
	DlgExposureTimeSet(QWidget* parent = nullptr);
	~DlgExposureTimeSet();

private:
	void build_ui();
	void build_connect();
public:
	void SetCamera();
	void ResetCamera();
private:
	Ui::DlgExposureTimeSetClass* ui;
private slots:
	void pbtn_exposureTimeValue_clicked();
	void pbtn_close_clicked();
};
