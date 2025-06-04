#pragma once

#include <QMainWindow>

#include"DraggableListWidget.h"

QT_BEGIN_NAMESPACE
namespace Ui { class PictureViewerThumbnailsClass; };
QT_END_NAMESPACE

class PictureViewerThumbnails : public QMainWindow
{
	Q_OBJECT
private:
	DraggableListWidget* _listWidget = nullptr;

public:
	PictureViewerThumbnails(QWidget *parent = nullptr);
	~PictureViewerThumbnails();
public:
	void setRootPath(const QString& rootPath);

	void setSize(const QSize& size);

private:
	void build_ui();
	void build_connect();
private:
	void loadImageList();
	void loadThumbnail(const QString& imagePath, QListWidgetItem* item);

private:
	Ui::PictureViewerThumbnailsClass *ui;
private:

	QString m_rootPath;
	QSize m_thumbnailSize{ 128, 128 };
	QStringList m_imageFiles;
private slots:
	void onThumbnailSelected();
};
