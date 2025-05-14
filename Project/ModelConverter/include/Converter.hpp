#pragma once

#include <QThread>

#include<QProcess>


class Converter
	: public QThread
{
	Q_OBJECT
public:
	void cancel();
public:
	QString inputFile;
	QString outputFile;
public:
	explicit Converter(QObject* parent = nullptr);

	~Converter() override;
private:
	QProcess* _processExportToEngine{ nullptr };
public:
	void run() override;
signals:
	void appRunLog(QString log);
	void finish();
public slots:
	void handleOutput();
	void handleError();
	void handleFinished(int exitCode, QProcess::ExitStatus exitStatus);

};
