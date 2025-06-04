#pragma once

#include <QMainWindow>
#include "ui_PictureViewerThumbnails.h"

QT_BEGIN_NAMESPACE
namespace Ui { class PictureViewerThumbnailsClass; };
QT_END_NAMESPACE

class PictureViewerThumbnails : public QMainWindow
{
	Q_OBJECT

public:
	PictureViewerThumbnails(QWidget *parent = nullptr);
	~PictureViewerThumbnails();

private:
	Ui::PictureViewerThumbnailsClass *ui;
};
