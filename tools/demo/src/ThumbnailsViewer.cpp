#include "ThumbnailsViewer.hpp"
#include <QVBoxLayout>
#include <QScrollBar>
#include <QFileInfo>
#include <QImageReader>
#include <QPixmap>
#include <QIcon>
#include <QPainter>

ThumbnailsViewer::ThumbnailsViewer(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    m_listWidget = new DraggableListWidget(this);
    m_listWidget->setViewMode(QListView::IconMode);
    m_listWidget->setResizeMode(QListView::Adjust);
    m_listWidget->setIconSize(m_thumbnailSize);
    m_listWidget->setMovement(QListView::Static);
    m_listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
    m_listWidget->setSpacing(8);
    layout->addWidget(m_listWidget);
    setLayout(layout);

    connect(m_listWidget, &QListWidget::itemSelectionChanged, this, &ThumbnailsViewer::onThumbnailSelected);
}

void ThumbnailsViewer::setRootPath(const QString& rootPath)
{
    m_rootPath = rootPath;
    loadImageList();
}

void ThumbnailsViewer::setSize(const QSize& size)
{
    m_thumbnailSize = size;
    m_listWidget->setIconSize(m_thumbnailSize);

    // 设置每个格子的大小，适当加padding
    QSize cellSize(m_thumbnailSize.width() + 16, m_thumbnailSize.height() + 32);
    for (int i = 0; i < m_listWidget->count(); ++i) {
        QListWidgetItem* item = m_listWidget->item(i);
        item->setSizeHint(cellSize);
        loadThumbnail(item->data(Qt::UserRole).toString(), item);
    }
}

void ThumbnailsViewer::setStatusBar(QStatusBar* statusBar)
{
    m_statusBar = statusBar;
}

void ThumbnailsViewer::wheelEvent(QWheelEvent* event)
{
    QWidget::wheelEvent(event);
}

void ThumbnailsViewer::onThumbnailSelected()
{
    if (!m_statusBar) return;
    QListWidgetItem* item = m_listWidget->currentItem();
    if (item) {
        QString path = item->data(Qt::UserRole).toString();
        m_statusBar->showMessage(path);
    }
}

void ThumbnailsViewer::loadImageList()
{
    m_listWidget->clear();
    m_imageFiles.clear();

    QDir dir(m_rootPath);
    QStringList nameFilters;
    // 支持绝大多数图片格式
    const auto formats = QImageReader::supportedImageFormats();
    for (const QByteArray& fmt : formats)
        nameFilters << "*." + fmt;

    QFileInfoList fileList = dir.entryInfoList(nameFilters, QDir::Files | QDir::NoSymLinks | QDir::Readable, QDir::Name);
    for (const QFileInfo& fileInfo : fileList) {
        m_imageFiles << fileInfo.absoluteFilePath();
        QListWidgetItem* item = new QListWidgetItem();
        item->setText(fileInfo.fileName());
        item->setData(Qt::UserRole, fileInfo.absoluteFilePath());
        item->setSizeHint(QSize(m_thumbnailSize.width() + 16, m_thumbnailSize.height() + 32));
        m_listWidget->addItem(item);
        loadThumbnail(fileInfo.absoluteFilePath(), item);
    }
}

void ThumbnailsViewer::loadThumbnail(const QString& imagePath, QListWidgetItem* item)
{
    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    QImage img = reader.read();
    if (!img.isNull()) {
        // 按比例缩放图片，最大边等于缩略图格子边长
        QImage scaledImg = img.scaled(m_thumbnailSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

        // 创建正方形透明底图
        QPixmap squarePixmap(m_thumbnailSize);
        squarePixmap.fill(Qt::transparent);

        // 计算居中位置
        QPainter painter(&squarePixmap);
        int x = (m_thumbnailSize.width() - scaledImg.width()) / 2;
        int y = (m_thumbnailSize.height() - scaledImg.height()) / 2;
        painter.drawImage(x, y, scaledImg);
        painter.end();

        item->setIcon(QIcon(squarePixmap));
    }
    else {
        // 加载失败时显示占位图标
        item->setIcon(QIcon());
    }
}