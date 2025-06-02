#pragma once

#include <QWidget>
#include <QListWidget>
#include <QDir>
#include <QMouseEvent>
#include<QScrollBar>


class ThumbnailListWidget : public QListWidget
{
    Q_OBJECT
public:
    explicit ThumbnailListWidget(QWidget* parent = nullptr)
        : QListWidget(parent), _dragging(false), _lastY(0) {
    }

protected:
    void mousePressEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::LeftButton) {
            _dragging = true;
            _lastY = event->globalY();
            setCursor(Qt::ClosedHandCursor);
        }
        QListWidget::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        if (_dragging) {
            int dy = event->globalY() - _lastY;
            _lastY = event->globalY();
            verticalScrollBar()->setValue(verticalScrollBar()->value() - dy);
        }
        QListWidget::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::LeftButton) {
            _dragging = false;
            unsetCursor();
        }
        QListWidget::mouseReleaseEvent(event);
    }

private:
    bool _dragging;
    int _lastY;
};

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
    ThumbnailListWidget* _listWidget; // 修改类型

    QString _rootPath;
    QSize _gridSize{ 250, 250 };      // 网格大小
    QSize _iconSize{ 100, 100 };      // 图标大小

    void updateIconSizeByGrid();    // 根据网格自动调整iconSize
};