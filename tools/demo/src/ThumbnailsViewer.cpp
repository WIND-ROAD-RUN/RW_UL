#include "ThumbnailsViewer.hpp"
#include <QHBoxLayout>
#include <QFileInfoList>
#include <QPixmap>
#include <QIcon>


ThumbnailsViewer::ThumbnailsViewer(QWidget* parent)
    : QWidget(parent)
    , _listWidget(new ThumbnailListWidget(this)) // 修改为自定义类
{
    auto* layout = new QHBoxLayout(this);
    layout->addWidget(_listWidget);
    _listWidget->setViewMode(QListWidget::IconMode);
    _listWidget->setResizeMode(QListWidget::Adjust);
    _listWidget->setSpacing(10);

    _listWidget->setDragEnabled(false);
    _listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
}

void ThumbnailsViewer::setRootPath(const QString& path)
{
    _rootPath = path;
    loadThumbnails(_rootPath);
}

void ThumbnailsViewer::setThumbnailGridSize(const QSize& gridSize)
{
    _gridSize = gridSize;
    updateIconSizeByGrid();
    _listWidget->setGridSize(_gridSize);
    _listWidget->setIconSize(_iconSize);
}

void ThumbnailsViewer::updateIconSizeByGrid()
{
    // 让iconSize比gridSize略小，留出空间显示文件名
    int iconW = std::max(32, _gridSize.width() - 20);
    int iconH = std::max(32, _gridSize.height() - 40);
    _iconSize = QSize(iconW, iconH);
}

void ThumbnailsViewer::loadThumbnails(const QString& path)
{
    _listWidget->clear();
    QDir dir(path);
    QStringList filters;
    filters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.gif";
    QFileInfoList fileList = dir.entryInfoList(filters, QDir::Files | QDir::NoSymLinks);

    for (const QFileInfo& fileInfo : fileList) {
        QPixmap pixmap(fileInfo.absoluteFilePath());
        QListWidgetItem* item = new QListWidgetItem(
            QIcon(pixmap.scaled(_iconSize, Qt::KeepAspectRatio, Qt::SmoothTransformation)),
            fileInfo.fileName());
        item->setToolTip(fileInfo.absoluteFilePath());
        _listWidget->addItem(item);
    }
}