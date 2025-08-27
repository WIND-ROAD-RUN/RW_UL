#include "rqw_ImageSaveEngineV1.hpp"

#include <iostream>
#include <QDir>

namespace rw {
	namespace rqw {
		QString imageFormatToString(rw::rqw::ImageSaveFormatV1 format)
		{
			switch (format) {
			case ImageSaveFormatV1::JPEG:
				return "jpg";
			case ImageSaveFormatV1::PNG:
				return "png";
			case ImageSaveFormatV1::BMP:
				return "bmp";
			default:
				return "jpg"; // 默认格式
			}
		}

		ImageSaveEngineV1::ImageSaveEngineV1(QObject* parent, int threadCount)
			: QObject(parent), stopFlag(false), threadCount(threadCount) {
		}

		ImageSaveEngineV1::~ImageSaveEngineV1()
		{
			stopEngine();
		}

		void ImageSaveEngineV1::pushImage(const ImageSaveInfoV1& image)
		{
			if (image.saveDirectoryPath.isEmpty())
			{
				return;
			}

			QMutexLocker locker(&mutex);
			// 如果策略为 SaveAllImg，则不限制队列大小
			if (savePolicy != ImageSaveEnginePolicyV1::SaveAllImg && saveQueue.size() >= maxQueueSize) {
				std::cerr << "Queue is full, dropping image." << std::endl;
				return;
			}
			saveQueue.enqueue(image);
			condition.wakeOne();
		}

		void ImageSaveEngineV1::stopEngine()
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

		void ImageSaveEngineV1::startEngine()
		{
			for (int i = 0; i < threadCount; ++i) {
				QThread* worker = QThread::create([this]() { this->processImages(); });
				workerThreads.push_back(worker);
				worker->start();
			}
		}

		void ImageSaveEngineV1::setSaveImgFormat(const ImageSaveFormatV1 format)
		{
			_saveImgFormat = format;
		}

		void ImageSaveEngineV1::processImages()
		{
			while (true) {
				QList<ImageSaveInfoV1> tasks;

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

		bool ImageSaveEngineV1::isAllImageSaved()
		{
			QMutexLocker locker(&mutex);
			return saveQueue.isEmpty();
		}

		void ImageSaveEngineV1::saveImage(const ImageSaveInfoV1& image)
		{
			QDir dir(image.saveDirectoryPath + "/");
			if (!dir.exists()) {
				dir.mkpath(".");
			}

			// 获取格式字符串
			QString formatStr = imageFormatToString(_saveImgFormat);

			// 构造文件名
			QString fileName = dir.filePath( image.name + "." + formatStr);

			// 检查策略
			if (savePolicy == ImageSaveEnginePolicyV1::MaxSaveImageNum) {
				auto& imageList = savedImages[image.saveDirectoryPath];
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

		void ImageSaveEngineV1::setSavePolicy(ImageSaveEnginePolicyV1 policy)
		{
			QMutexLocker locker(&mutex);
			savePolicy = policy;
		}

		void ImageSaveEngineV1::setMaxSaveImageNum(int maxNum)
		{
			QMutexLocker locker(&mutex);
			maxSaveImageNum = maxNum;
		}

		void ImageSaveEngineV1::setSaveImgQuality(int quality)
		{
			QMutexLocker locker(&mutex);
			saveImgQuality = quality;
		}
	}
}