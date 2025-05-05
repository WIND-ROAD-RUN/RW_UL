#pragma once

#include <QDialog>
#include "ui_LoadingDialog.h"

QT_BEGIN_NAMESPACE
namespace Ui { class LoadingDialogClass; };
QT_END_NAMESPACE

class LoadingDialog : public QDialog
{
	Q_OBJECT

public:
	explicit LoadingDialog(QWidget* parent = nullptr)
		: QDialog(parent) {
		setWindowTitle("加载中...");
		setWindowFlags(windowFlags() & ~Qt::WindowCloseButtonHint); // 禁用关闭按钮
		setModal(true);

		QVBoxLayout* layout = new QVBoxLayout(this);
		label = new QLabel("正在加载...", this);
		progressBar = new QProgressBar(this);
		progressBar->setRange(0, 0); // 设置为不确定模式

		layout->addWidget(label);
		layout->addWidget(progressBar);
		setLayout(layout);
	}

	void updateMessage(const QString& message) {
		label->setText(message);
	}

private:
	QLabel* label;
	QProgressBar* progressBar;

private:
	Ui::LoadingDialogClass* ui;
};
