#pragma once

#include <QDir>
#include <QMainWindow>
#include <QStandardItemModel>

#include"DraggableListWidget.h"
#include"LoadingDialog.h"

QT_BEGIN_NAMESPACE
namespace Ui { class PictureViewerThumbnailsClass; };
QT_END_NAMESPACE

class PictureViewerThumbnails : public QMainWindow
{
	Q_OBJECT
private:
	QHash<QString, QStringList> m_categoryImageCache; // Ä¿Â¼Â·¾¶->Í¼Æ¬Â·¾¶ÁÐ±í»º´æ
	QHash<QString, QPixmap> m_thumbnailCache;         // Í¼Æ¬Â·¾¶->ËõÂÔÍ¼
private:
	LoadingDialog* _loadingDialog = nullptr;
public:
	PictureViewerThumbnails(QWidget *parent = nullptr);
	~PictureViewerThumbnails();
public:
	void setRootPath(const QString& rootPath);

	void setSize(const QSize& size);
protected:
	void showEvent(QShowEvent* event) override;
private:
	void build_ui();
	void build_connect();
private:
	void loadImageList();
	void loadThumbnail(const QString& imagePath, QListWidgetItem* item);
	void updateCategoryList();
	void addSubFolders(const QDir& parentDir, QStandardItem* parentItem);
	QModelIndex findFirstDeepestIndex(QStandardItemModel* model);
	QModelIndex findDeepestChild(QStandardItem* parentItem);
private:
	void preloadAllCategoryImages(const QDir& dir);

private:
	Ui::PictureViewerThumbnailsClass *ui;
private:

	QString m_rootPath;
	QSize m_thumbnailSize{ 128, 128 };
	QStringList m_imageFiles;
private:
	QStandardItemModel* _categoryModel;
	DraggableListWidget* _listWidget = nullptr;
private slots:
	void onThumbnailSelected();
	void pbtn_exit_clicked();
	void pbtn_deleteTotal_clicked();
	void pbtn_delete_clicked();
	void pbtn_prePicture_clicked();
	void pbtn_nextPicture_clicked();
private slots:
	void onThumbnailDoubleClicked(QListWidgetItem* item);
};
