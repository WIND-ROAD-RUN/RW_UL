#pragma once

#include <QThread>
#include <QVector>
#include <QString>
#include <QDebug>
#include <QImage>
#include <QPixmap>

#include"ime_ModelEngineFactory.h"

class AutomaticAnnotationThread : public QThread {
    Q_OBJECT

public:
    explicit AutomaticAnnotationThread(const QVector<QString>& imagePaths, QObject* parent = nullptr);

signals:
    void imageProcessed(QString imagePath,QPixmap pixmap);

protected:
    void run() override;
public:
    rw::ModelEngineConfig config;

private:
    QVector<QString> m_imagePaths;
};