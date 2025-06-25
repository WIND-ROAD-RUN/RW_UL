#include "rqw_ImageSaveEngine.h"

#include <iostream>
#include <QDir>

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

		void ImageSaveEngine::pushImage(const ImageInfo& image)
		{
			QMutexLocker locker(&mutex);
			if (saveQueue.size() >= maxQueueSize) {
				std::cerr << "Queue is full, dropping image." << std::endl;
				return;
			}
			saveQueue.enqueue(image);
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
				QList<ImageInfo> tasks;

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
					saveImage(task);
				}
			}
		}

		void ImageSaveEngine::saveImage(const ImageInfo& image)
		{
			QDir dir(rootPath + "/" + image.classify);
			if (!dir.exists()) {
				dir.mkpath(".");
			}

			// 构造文件名
			QString fileName = dir.filePath(image.classify + image.time + ".jpg");

			// 检查策略
			if (savePolicy == ImageSaveEnginePolicy::MaxSaveImageNum) {
				auto& imageList = savedImages[image.classify];
				if (imageList.size() >= maxSaveImageNum) {
					// 删除最旧的图片
					QString oldestFile = imageList.front();
					imageList.erase(imageList.begin());
					QFile::remove(oldestFile);
				}
				imageList.push_back(fileName);
			}

			// 保存图片
			if (!image.image.save(fileName)) {
				std::cerr << "Failed to save image: " << fileName.toStdString() << std::endl;
			}
		}

		void ImageSaveEngine::setSavePolicy(ImageSaveEnginePolicy policy)
		{
			QMutexLocker locker(&mutex);
			savePolicy = policy;
		}

		void ImageSaveEngine::setMaxSaveImageNum(int maxNum)
		{
			QMutexLocker locker(&mutex);
			maxSaveImageNum = maxNum;
		}
	}
}