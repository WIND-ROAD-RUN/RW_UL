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
		class ImageSaveEngineV1 : public QObject {
			Q_OBJECT

		public:
			ImageSaveEngineV1(QObject* parent = nullptr, int threadCount = 4);

			~ImageSaveEngineV1();

			void pushImage(const ImageSaveInfoV1& image);

			void stopCom();

			void buildCom();

			void setSavePolicy(ImageSaveEnginePolicyV1 policy);

			void setMaxSaveImageNum(int maxNum);

		private:
			int saveImgQuality = 99;
		public:
			void setSaveImgQuality(int quality);
		private:
			ImageSaveFormatV1 _saveImgFormat = ImageSaveFormatV1::JPEG;
		public:
			void setSaveImgFormat(ImageSaveFormatV1 format);
		protected:
			void processImages();
		public:
			bool isAllImageSaved();
		private:
			void saveImage(const ImageSaveInfoV1& image);

			QQueue<ImageSaveInfoV1> saveQueue;
			QMutex mutex;
			QWaitCondition condition;
			std::atomic<bool> stopFlag;

			const int maxQueueSize = 80;
			const int batchSize = 20;
			int threadCount;
			std::vector<QThread*> workerThreads;

			ImageSaveEnginePolicyV1 savePolicy = ImageSaveEnginePolicyV1::Normal;
			int maxSaveImageNum = 50;
			std::map<QString, std::vector<QString>> savedImages;
		};
	}
}