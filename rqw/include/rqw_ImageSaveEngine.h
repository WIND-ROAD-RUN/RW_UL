#pragma once
#include <QImage>
#include <QString>
#include <QDir>
#include <QMutex>
#include <QThread>
#include <QQueue>
#include <QWaitCondition>
#include <QDateTime>
#include <atomic>
#include <iostream>
#include <vector>

namespace rw {
	namespace rqw {
		class ImageSaveEngine : public QThread {
			Q_OBJECT

		public:
			ImageSaveEngine(QObject* parent = nullptr, int threadCount = 4);

			~ImageSaveEngine();

			// 设置根路径
			void setRootPath(const QString& rootPath);

			QString getRootPath();

			// 修改后的 pushImage 方法
			void pushImage(const QImage& image, const QString& classify, const QString& namePrefix);

			// 停止线程
			void stop();

			// 启动线程池
			void startEngine();

		protected:
			void processImages();

		private:
			void saveImage(const QImage& image, const QString& classifyWithPrefix);

			QString rootPath;
			QQueue<QPair<QImage, QString>> saveQueue;
			QMutex mutex;
			QWaitCondition condition;
			std::atomic<bool> stopFlag;

			const int maxQueueSize = 80; // 队列最大容量
			const int batchSize = 20;     // 每次批量保存的图片数量
			int threadCount;              // 消费者线程数量
			std::vector<QThread*> workerThreads;
		};
	}
}