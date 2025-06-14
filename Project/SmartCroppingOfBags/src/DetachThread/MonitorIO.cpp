#include "MonitorIO.h"

MonitorIOSmartCroppingOfBags::MonitorIOSmartCroppingOfBags(QObject* parent)
{

}

MonitorIOSmartCroppingOfBags::~MonitorIOSmartCroppingOfBags()
{
	stopThread();
	wait(); // 等待线程安全退出
}

void MonitorIOSmartCroppingOfBags::startThread()
{
	running = true;
	if (!isRunning()) {
		start(); // 启动线程
	}
}

void MonitorIOSmartCroppingOfBags::stopThread()
{
	running = false; // 停止线程
}

void MonitorIOSmartCroppingOfBags::run()
{
	auto& motion = GlobalStructDataSmartCroppingOfBags::getInstance().motion;
	while (running) {
		// 这里可以添加监控IO的逻辑

		//运动控制卡获得当前IO状态
		bool nowstate = motion->GetIOIn(0);

		// 上升延
		if (nowstate == true && state == false)
		{
			//求5个袋子的平均袋长
			//1通过像素求
			//2通过脉冲去求

		}

		QThread::usleep(10);
	}
	// 线程结束时可以进行清理工作
	qDebug() << "MonitorIOSmartCroppingOfBags thread stopped.";
}
