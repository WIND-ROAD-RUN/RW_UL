#pragma once

#include <QThread>
#include <atomic>

class DetachUtiltyThreadSmartCroppingOfBags : public QThread
{
	Q_OBJECT
public:
	std::atomic_bool isProcessing{ false };
	std::atomic_bool isProcessFinish{ false };
public:
	explicit DetachUtiltyThreadSmartCroppingOfBags(QObject* parent = nullptr);

	~DetachUtiltyThreadSmartCroppingOfBags() override;

	void startThread();

	void stopThread();

protected:
	void run() override;
private:
	void getMaiChongXinhao(size_t s);
signals:
	void updateCurrentPulse(double pulse);
private:
	std::atomic<bool> running; 
};
