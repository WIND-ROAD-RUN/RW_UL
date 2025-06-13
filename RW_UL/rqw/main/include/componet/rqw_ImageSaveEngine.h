#pragma once
#include <QImage>
#include <QString>
#include <QMutex>
#include <QThread>
#include <QQueue>
#include <QWaitCondition>
#include <QDateTime>
#include <atomic>
#include <vector>

namespace rw {
	namespace rqw {
		struct ImageInfo
		{
		public:
			QImage image;
			QString classify;
		public:
			QString time;
		public:
			ImageInfo(const QImage & image)
			{
				this->image = image;
				QDateTime currentTime = QDateTime::currentDateTime();
				this->time = currentTime.toString("yyyyMMddhhmmsszzz"); // 年月日时分秒毫秒
			}
		};


		class ImageSaveEngine : public QThread {
			Q_OBJECT

		public:
			ImageSaveEngine(QObject* parent = nullptr, int threadCount = 4);

			~ImageSaveEngine();

			// 设置根路径
			void setRootPath(const QString& rootPath);

			QString getRootPath();

			// 修改后的 pushImage 方法
			void pushImage(const ImageInfo& image);

			// 停止线程
			void stop();

			// 启动线程池
			void startEngine();

		protected:
			void processImages();

		private:
			void saveImage(const ImageInfo& image);

			QString rootPath;
			QQueue<ImageInfo> saveQueue;
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