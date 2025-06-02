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
	DlgProductScore(QWidget *parent = nullptr);
	~DlgProductScore();

public:
	void build_ui();
	void read_config();

private:
	Ui::DlgProductScoreClass *ui;
};

