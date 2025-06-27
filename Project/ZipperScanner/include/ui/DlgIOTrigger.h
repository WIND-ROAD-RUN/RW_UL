#pragma once

#include <QDialog>
#include "ui_DlgIOTrigger.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgIOTriggerClass; };
QT_END_NAMESPACE

class DlgIOTrigger : public QDialog
{
	Q_OBJECT

public:
	DlgIOTrigger(QWidget *parent = nullptr);
	~DlgIOTrigger();

private:
	void build_connect();

private slots:
	void pbtn_close_clicked();

	void btn_shoudongladai_pressed();
	void btn_shoudongladai_released();
	void btn_shoudongchongkong_clicked();
	void btn_tuoji_clicked();
	void btn_xiangjichufa_clicked();

private:
	Ui::DlgIOTriggerClass *ui;
};

