#include"rqw_ImageSaveEngine.h"

namespace rw {
	namespace rqw {
		ImageSaveEngine::ImageSaveEngine(QObject* parent, int threadCount)
			: QThread(parent), stopFlag(false), threadCount(threadCount) {
		}

		ImageSaveEngine::~ImageSaveEngine()
		{
			stop();
		}

		void ImageSaveEngine::setRootPath(const QString& rootPath)
		{
			QMutexLocker locker(&mutex);
			this->rootPath = rootPath;
			QDir dir(rootPath);
			if (!dir.exists()) {
				dir.mkpath(".");
			}
		}

		QString ImageSaveEngine::getRootPath()
		{
			QMutexLocker locker(&mutex);
			if (rootPath.isEmpty()) {
				std::cerr << "Root path is not set." << std::endl;
				return QString();
			}
			return rootPath;
		}

		void ImageSaveEngine::pushImage(const QImage& image, const QString& classify, const QString& namePrefix)
		{
			QMutexLocker locker(&mutex);
			if (saveQueue.size() >= maxQueueSize) {
				std::cerr << "Queue is full, dropping image." << std::endl;
				return;
			}
			saveQueue.enqueue({ image, classify });
			condition.wakeOne();
		}

		void ImageSaveEngine::stop()
		{
			{
				QMutexLocker locker(&mutex);
				stopFlag = true;
				condition.wakeAll();
			}
			for (auto& thread : workerThreads) {
				thread->wait();
				delete thread;
			}
			workerThreads.clear();
		}

		void ImageSaveEngine::startEngine()
		{
			for (int i = 0; i < threadCount; ++i) {
				QThread* worker = QThread::create([this]() { this->processImages(); });
				workerThreads.push_back(worker);
				worker->start();
			}
		}

		void ImageSaveEngine::processImages()
		{
			while (true) {
				QList<QPair<QImage, QString>> tasks;

				{
					QMutexLocker locker(&mutex);
					if (saveQueue.isEmpty() && !stopFlag) {
						condition.wait(&mutex);
					}

					if (stopFlag && saveQueue.isEmpty()) {
						break;
					}

					// 批量取出任务
					while (!saveQueue.isEmpty() && tasks.size() < batchSize) {
						tasks.append(saveQueue.dequeue());
					}
				}

				// 保存图片
				for (const auto& task : tasks) {
					saveImage(task.first, task.second);
				}
			}
		}

		void ImageSaveEngine::saveImage(const QImage& image, const QString& classifyWithPrefix)
		{
			QDir dir(rootPath + "/" + classifyWithPrefix);
			if (!dir.exists()) {
				dir.mkpath(".");
			}

			// 获取当前时间
			QDateTime currentTime = QDateTime::currentDateTime();
			QString timestamp = currentTime.toString("yyyyMMddhhmmsszzz"); // 年月日时分秒毫秒

			// 构造文件名
			QString fileName = dir.filePath(classifyWithPrefix + timestamp + ".png");

			// 保存图片
			if (!image.save(fileName)) {
				std::cerr << "Failed to save image: " << fileName.toStdString() << std::endl;
			}
		}
	}
}