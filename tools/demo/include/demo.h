#pragma once

#include <QMainWindow>
#include "ui_demo.h"

QT_BEGIN_NAMESPACE
namespace Ui { class demoClass; };
QT_END_NAMESPACE

#include"rqw_CameraObjectThread.hpp"

#include"ime_ModelEngineFactory.h"

class demo : public QMainWindow
{
	Q_OBJECT

public:
	demo(QWidget *parent = nullptr);
	~demo();
private:
	rw::rqw::CameraPassiveThread m_cameraThread;
	std::unique_ptr<rw::ModelEngine> engine;
private:
	Ui::demoClass *ui;
private slots:
	void displayImg(cv::Mat frame);
};
