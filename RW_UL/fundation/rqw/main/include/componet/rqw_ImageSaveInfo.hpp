#pragma once

#include <QImage>
#include <QString>
#include <QDateTime>

namespace rw
{
	namespace rqw
	{
		enum class ImageSaveEnginePolicyV1
		{
			Normal,
			MaxSaveImageNum,
			SaveAllImg
		};

		enum class ImageSaveFormatV1
		{
			JPEG,
			PNG,
			BMP
		};

		struct ImageSaveInfoV1
		{
		public:
			QImage image;
		public:
			QString name;
			QString saveDirectoryPath{};
		public:
			ImageSaveInfoV1(const QImage& image)
			{
				this->image = image;
				QDateTime currentTime = QDateTime::currentDateTime();
				this->name = currentTime.toString("yyyyMMddhhmmsszzz");
			}

			ImageSaveInfoV1(QImage&& image)
			{
				this->image = std::move(image);
				QDateTime currentTime = QDateTime::currentDateTime();
				this->name = currentTime.toString("yyyyMMddhhmmsszzz");
			}
		};

		QString imageFormatToString(rw::rqw::ImageSaveFormatV1 format);
	}

}
