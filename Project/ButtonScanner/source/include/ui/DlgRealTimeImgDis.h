#pragma once

#include <QDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class DlgRealTimeImgDisClass; };
QT_END_NAMESPACE

class DlgRealTimeImgDis : public QDialog
{
	Q_OBJECT

public:
	DlgRealTimeImgDis(QWidget *parent = nullptr);
	~DlgRealTimeImgDis();
private:
	void build_ui();
	void build_connect();
private:
	bool* _isShow;
public:
	void setMonitorValue(bool * isShow);
public:
	void setGboxTitle(const QString & title);
protected:
	void showEvent(QShowEvent* event) override;
private:
	Ui::DlgRealTimeImgDisClass *ui;
public slots:
	void pbtn_exit_clicked();
};
