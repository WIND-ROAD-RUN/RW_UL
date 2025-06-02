#pragma once

#include <QWidget>
#include <QListWidget>
#include <QDir>

class ThumbnailsViewer : public QWidget
{
    Q_OBJECT
public:
    explicit ThumbnailsViewer(QWidget* parent = nullptr);

    void setRootPath(const QString& path);
    void setThumbnailGridSize(const QSize& gridSize); // 新增：设置网格大小并自动调整缩略图

protected:
    void loadThumbnails(const QString& path);

private:
    QListWidget* _listWidget;
    QString _rootPath;
    QSize _gridSize{ 250, 250 };      // 网格大小
    QSize _iconSize{ 100, 100 };      // 图标大小

    void updateIconSizeByGrid();    // 根据网格自动调整iconSize
};