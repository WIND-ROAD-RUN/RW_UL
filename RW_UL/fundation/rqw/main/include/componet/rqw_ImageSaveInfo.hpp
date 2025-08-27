#pragma once

#include <QImage>
#include <QString>
#include <QDateTime>

namespace rw
{
	namespace rqw
	{
		enum class ImageSaveEnginePolicyRefactor
		{
			Normal,
			MaxSaveImageNum,
			SaveAllImg
		};

		enum class ImageSaveFormatRefactor
		{
			JPEG,
			PNG,
			BMP
		};

		struct ImageSaveInfoRefactor
		{
		public:
			QImage image;
			QString classify;
		public:
			QString time;
			QString saveDirectoryPath{};
		public:
			ImageSaveInfoRefactor(const QImage& image)
			{
				this->image = image;
				QDateTime currentTime = QDateTime::currentDateTime();
				this->time = currentTime.toString("yyyyMMddhhmmsszzz");
			}
		};

		QString imageFormatToString(rw::rqw::ImageSaveFormatRefactor format);
	}

}
