#pragma once

#include <QDialog>
#include "ui_PicturesPainterVersionDunDai.h"

QT_BEGIN_NAMESPACE
namespace Ui { class PicturesPainterVersionDunDaiClass; };
QT_END_NAMESPACE

class PicturesPainterVersionDunDai : public QDialog
{
	Q_OBJECT

public:
	PicturesPainterVersionDunDai(QWidget *parent = nullptr);
	~PicturesPainterVersionDunDai();

private:
	Ui::PicturesPainterVersionDunDaiClass *ui;
};

