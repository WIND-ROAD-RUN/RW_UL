//#pragma once
//#include <QThread>
//#include <QDebug>
//#include <atomic>
//#include "SetConfig.hpp"
//
//class CameraAndCardStateThreadSCOB : public QThread
//{
//	Q_OBJECT
//private:
//	static size_t runtimeCounts;
//
//public:
//	explicit CameraAndCardStateThreadSCOB(QObject* parent = nullptr);
//
//	~CameraAndCardStateThreadSCOB() override;
//
//	void startThread();
//
//	void stopThread();
//
//	cdm::SetConfig& _dlgProductSet;
//
//protected:
//	void run() override;
//private:
//	void check_cameraState();
//	void check_cameraState1();
//	void check_cameraState2();
//private:
//	void check_cardState();
//
//signals:
//	void updateCameraLabelState(int cameraIndex, bool state);
//	void updateCardLabelState(bool state);
//	void addWarningInfo(QString message, bool updateTimestampIfSame, int redDuration);
//
//signals:
//	void buildCamera1();
//	void buildCamera2();
//
//	void destroyCamera1();
//	void destroyCamera2();
//
//	void startMonitor1();
//	void startMonitor2();
//
//private:
//	std::atomic<bool> running; // 使用原子变量保证线程安全
//};