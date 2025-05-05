#pragma once
#include <QThread>
#include <QDebug>
#include <atomic>

class MonitorCameraAndCardStateThread : public QThread
{
	Q_OBJECT
private:
	static size_t runtimeCounts;
public:
	explicit MonitorCameraAndCardStateThread(QObject* parent = nullptr);

	~MonitorCameraAndCardStateThread() override;

	void startThread();

	void stopThread();

protected:
	void run() override;
private:
	void check_cameraState();
	void check_cameraState1();
	void check_cameraState2();
	void check_cameraState3();
	void check_cameraState4();
private:
	void check_cardState();

signals:
	void updateCameraLabelState(int cameraIndex, bool state);
	void updateCardLabelState(bool state);
	void addWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);

signals:
	void buildCamera1();
	void buildCamera2();
	void buildCamera3();
	void buildCamera4();

	void destroyCamera1();
	void destroyCamera2();
	void destroyCamera3();
	void destroyCamera4();

	void startMonitor1();
	void startMonitor2();
	void startMonitor3();
	void startMonitor4();

private:
	std::atomic<bool> running; // 使用原子变量保证线程安全
};