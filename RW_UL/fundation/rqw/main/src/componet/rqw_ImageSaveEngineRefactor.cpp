#include "rqw_ImageSaveEngineRefactor.hpp"

#include <iostream>
#include <QDir>

namespace rw {
	namespace rqw {
		QString imageFormatToString(rw::rqw::ImageSaveFormatRefactor format)
		{
			switch (format) {
			case ImageSaveFormatRefactor::JPEG:
				return "jpg";
			case ImageSaveFormatRefactor::PNG:
				return "png";
			case ImageSaveFormatRefactor::BMP:
				return "bmp";
			default:
				return "jpg"; // 默认格式
			}
		}

		ImageSaveEngineRefactor::ImageSaveEngineRefactor(QObject* parent, int threadCount)
			: QObject(parent), stopFlag(false), threadCount(threadCount) {
		}

		ImageSaveEngineRefactor::~ImageSaveEngineRefactor()
		{
			stopEngine();
		}

		void ImageSaveEngineRefactor::setRootPath(const QString& rootPath)
		{
			QMutexLocker locker(&mutex);
			this->rootPath = rootPath;
			QDir dir(rootPath);
			if (!dir.exists()) {
				dir.mkpath(".");
			}
		}

		QString ImageSaveEngineRefactor::getRootPath()
		{
			QMutexLocker locker(&mutex);
			if (rootPath.isEmpty()) {
				std::cerr << "Root path is not set." << std::endl;
				return QString();
			}
			return rootPath;
		}

		void ImageSaveEngineRefactor::pushImage(const ImageSaveInfoRefactor& image)
		{
			QMutexLocker locker(&mutex);
			// 如果策略为 SaveAllImg，则不限制队列大小
			if (savePolicy != ImageSaveEnginePolicyRefactor::SaveAllImg && saveQueue.size() >= maxQueueSize) {
				std::cerr << "Queue is full, dropping image." << std::endl;
				return;
			}
			saveQueue.enqueue(image);
			condition.wakeOne();
		}

		void ImageSaveEngineRefactor::stopEngine()
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

		void ImageSaveEngineRefactor::startEngine()
		{
			for (int i = 0; i < threadCount; ++i) {
				QThread* worker = QThread::create([this]() { this->processImages(); });
				workerThreads.push_back(worker);
				worker->start();
			}
		}

		void ImageSaveEngineRefactor::setSaveImgFormat(const ImageSaveFormatRefactor format)
		{
			_saveImgFormat = format;
		}

		void ImageSaveEngineRefactor::processImages()
		{
			while (true) {
				QList<ImageSaveInfoRefactor> tasks;

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

					// 保存图片
					for (const auto& task : tasks) {
						saveImage(task);
					}
				}


			}
		}

		bool ImageSaveEngineRefactor::isAllImageSaved()
		{
			QMutexLocker locker(&mutex);
			return saveQueue.isEmpty();
		}

		void ImageSaveEngineRefactor::saveImage(const ImageSaveInfoRefactor& image)
		{
			QDir dir(rootPath + "/" + image.classify);
			if (!dir.exists()) {
				dir.mkpath(".");
			}

			// 获取格式字符串
			QString formatStr = imageFormatToString(_saveImgFormat);

			// 构造文件名
			QString fileName = dir.filePath(image.classify + image.time + "." + formatStr);

			// 检查策略
			if (savePolicy == ImageSaveEnginePolicyRefactor::MaxSaveImageNum) {
				auto& imageList = savedImages[image.classify];
				if (imageList.size() >= maxSaveImageNum) {
					QString oldestFile = imageList.front();
					imageList.erase(imageList.begin());
					QFile::remove(oldestFile);
				}
				imageList.push_back(fileName);
			}

			// 保存图片
			if (!image.image.save(fileName, formatStr.toUtf8().constData(), saveImgQuality)) {
				std::cerr << "Failed to save image: " << fileName.toStdString() << std::endl;
			}
		}

		void ImageSaveEngineRefactor::setSavePolicy(ImageSaveEnginePolicyRefactor policy)
		{
			QMutexLocker locker(&mutex);
			savePolicy = policy;
		}

		void ImageSaveEngineRefactor::setMaxSaveImageNum(int maxNum)
		{
			QMutexLocker locker(&mutex);
			maxSaveImageNum = maxNum;
		}

		void ImageSaveEngineRefactor::setSaveImgQuality(int quality)
		{
			QMutexLocker locker(&mutex);
			saveImgQuality = quality;
		}
	}
}