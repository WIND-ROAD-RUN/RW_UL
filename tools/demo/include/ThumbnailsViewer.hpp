#pragma once

#include <QString>
#include <QWidget>
#include <QDir>
#include <QStatusBar>
#include <QImageReader>
#include "DraggableListWidget.hpp"

class ThumbnailsViewer : public QWidget
{
	Q_OBJECT

public:
	explicit ThumbnailsViewer(QWidget* parent = nullptr);

	// 设置图片根路径
	void setRootPath(const QString& rootPath);

	// 设置缩略图大小
	void setSize(const QSize& size);

	// 设置状态栏
	void setStatusBar(QStatusBar* statusBar);

protected:
	void wheelEvent(QWheelEvent* event) override;

private slots:
	// 选中缩略图时显示路径
	void onThumbnailSelected();

private:
	void loadImageList();
	void loadThumbnail(const QString& imagePath, QListWidgetItem* item);

	QString m_rootPath;
	QSize m_thumbnailSize{ 128, 128 };
	DraggableListWidget1* m_listWidget;
	QStatusBar* m_statusBar{ nullptr };
	QStringList m_imageFiles;
};