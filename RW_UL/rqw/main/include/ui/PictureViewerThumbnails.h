#pragma once

#include <QDir>
#include <QMainWindow>
#include <QStandardItemModel>

#include"DraggableListWidget.h"
#include"LoadingDialog.h"
#include"PictureViewerUtilty.h"

#include<QListWidgetItem>

QT_BEGIN_NAMESPACE
namespace Ui { class PictureViewerThumbnailsClass; };
QT_END_NAMESPACE

namespace rw
{
	namespace dsl
	{
		template <typename Key, typename Value>
		class CacheFIFO;
	}
}

class PictureViewerThumbnails : public QMainWindow
{ 
	Q_OBJECT
private:
	QVector<QString> disCacheImagePaths;
private:
	std::unique_ptr<rw::dsl::CacheFIFO<QString, QPixmap>> _thumbnailCache;
private:
	QHash<QString, QStringList> m_categoryImageCache; 
	/*QHash<QString, QPixmap> m_thumbnailCache;  */       
private:
	LoadingDialog* _loadingDialog = nullptr;
	PictureViewerUtilty* pictureViewerUtilty=nullptr;
public:
	PictureViewerThumbnails(QWidget *parent = nullptr);
	~PictureViewerThumbnails() override;
public:
	void setThumbnailCacheCapacity(size_t capacity);
private:
	size_t _thumbnailCacheCapacity{1000};
public:
	void setRootPath(const QString& rootPath);

	void setSize(const QSize& size);
	void setSizeRange(const QSize & sizeSmall, const QSize& sizeBig);
private:
	QSize small{100,100};
	QSize big{ 500,500 };
signals:
	void viewerClosed();
protected:
	void showEvent(QShowEvent* event) override;
private:
	void build_ui();
	void build_connect();
private:
	void loadImageList();
	bool loadThumbnail(const QString& imagePath, QListWidgetItem* item);
	void loadThumbnailDisCache();
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
	void pbtn_bigger_clicked();
	void pbtn_smaller_clicked();
private slots:
	void onThumbnailDoubleClicked(QListWidgetItem* item);
};
