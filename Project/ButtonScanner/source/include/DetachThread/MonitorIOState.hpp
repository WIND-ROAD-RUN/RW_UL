#pragma once

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

class MonitorIOStateThread : public QThread
{
	Q_OBJECT
public:
	explicit MonitorIOStateThread(QObject* parent = nullptr);

	// set thraed state true=running false=sleep
	void setRunning(bool running);

	// destroy thread
	void destroyThread();

protected:
	void run() override;
private:
	void monitorDIState();
	void monitorDOState();
signals:
	void DIState(int index, bool state);
	void DOState(int index, bool state);
private:
	QMutex m_mutex;
	QWaitCondition m_waitCond;
	bool m_running{ false };
	bool m_exit{ false };
};