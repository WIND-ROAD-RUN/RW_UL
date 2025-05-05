#pragma once

#include <QDialog>
#include "ui_DlgHideScoreSet.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgHideScoreSetClass; };
QT_END_NAMESPACE

class DlgHideScoreSet : public QDialog
{
	Q_OBJECT

public:
	DlgHideScoreSet(QWidget* parent = nullptr);
	~DlgHideScoreSet();
private:
	void build_ui();
	void build_connect();

private:
	void readConfig();

private:
	Ui::DlgHideScoreSetClass* ui;

private slots:
	void pbtn_close_clicked();
	void pbtn_outsideDiameterScore_clicked();
	void pbtn_forAndAgainstScore_clicked();
};
