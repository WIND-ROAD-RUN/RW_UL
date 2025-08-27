#pragma once

#include"rqw_ImageSaveInfo.hpp"


#include <QMutex>
#include <QThread>
#include <QQueue>
#include <QWaitCondition>

#include <atomic>
#include <vector>
#include <map>


namespace rw {
	namespace rqw {
		class ImageSaveEngineRefactor : public QObject {
			Q_OBJECT

		public:
			ImageSaveEngineRefactor(QObject* parent = nullptr, int threadCount = 4);

			~ImageSaveEngineRefactor();

			void setRootPath(const QString& rootPath);

			QString getRootPath();

			void pushImage(const ImageSaveInfoRefactor& image);

			void stopEngine();

			void startEngine();

			void setSavePolicy(ImageSaveEnginePolicyRefactor policy);

			void setMaxSaveImageNum(int maxNum);

		private:
			int saveImgQuality = 99;
		public:
			void setSaveImgQuality(int quality);
		private:
			ImageSaveFormatRefactor _saveImgFormat = ImageSaveFormatRefactor::JPEG;
		public:
			void setSaveImgFormat(ImageSaveFormatRefactor format);
		protected:
			void processImages();
		public:
			bool isAllImageSaved();
		private:
			void saveImage(const ImageSaveInfoRefactor& image);

			QString rootPath;
			QQueue<ImageSaveInfoRefactor> saveQueue;
			QMutex mutex;
			QWaitCondition condition;
			std::atomic<bool> stopFlag;

			const int maxQueueSize = 80;
			const int batchSize = 20;
			int threadCount;
			std::vector<QThread*> workerThreads;

			ImageSaveEnginePolicyRefactor savePolicy = ImageSaveEnginePolicyRefactor::Normal;
			int maxSaveImageNum = 50;
			std::map<QString, std::vector<QString>> savedImages;
		};
	}
}