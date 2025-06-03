#pragma once

#include <QDialog>
#include "ui_DlgProductScore.h"

QT_BEGIN_NAMESPACE
namespace Ui { class DlgProductScoreClass; };
QT_END_NAMESPACE

class DlgProductScore : public QDialog
{
	Q_OBJECT

public:
	DlgProductScore(QWidget* parent = nullptr);
	~DlgProductScore();

public:
	void build_ui();
	void read_config();
	void build_connect();

private slots:
	void pbtn_close_clicked();

	void rbtn_queyaEnable_checked();
	void ptn_queyaSimilarity_clicked();
	void ptn_queyaArea_clicked();
	void rbtn_tangshangEnable_checked();
	void ptn_tangshangSimilarity_clicked();
	void ptn_tangshangArea_clicked();
	void rbtn_zangwuEnable_checked();
	void ptn_zangwuSimilarity_clicked();
	void ptn_zangwuArea_clicked();



private:
	Ui::DlgProductScoreClass* ui;
};

