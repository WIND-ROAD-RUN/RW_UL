#pragma once

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class PictureViewerUtiltyClass; };
QT_END_NAMESPACE

class PictureViewerUtilty : public QMainWindow
{
	Q_OBJECT

public:
	PictureViewerUtilty(QWidget *parent = nullptr);
	~PictureViewerUtilty();
private:
	void build_ui();
	void build_connect();
protected:
	void showEvent(QShowEvent* event) override;
public:
	void setImgPath(const QString & imgPath);
private:
	QString path;
private:
	Ui::PictureViewerUtiltyClass *ui;
private slots:
	void pbtn_exit_clicked();
};
