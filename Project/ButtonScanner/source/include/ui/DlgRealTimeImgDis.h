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
	int* _currentDisImgIndex;
public:
	void setMonitorValue(bool * isShow);
	void setMonitorDisImgIndex(int * index);
public:
	void setGboxTitle(const QString & title);
protected:
	void showEvent(QShowEvent* event) override;
public:
	void setShowImg(const QPixmap &image);
private:
	Ui::DlgRealTimeImgDisClass *ui;
public:
	void updateTitle(int index);
public slots:
	void pbtn_exit_clicked();
	void pbtn_nextWork_clicked();
	void pbtn_preWork_clicked();
};
