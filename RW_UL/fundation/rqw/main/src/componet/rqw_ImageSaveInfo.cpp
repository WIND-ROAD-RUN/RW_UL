#include"rqw_ImageSaveInfo.hpp"


namespace rw
{
	namespace rqw
	{
		QString imageFormatToString(rw::rqw::ImageSaveFormatRefactor format)
		{
			switch (format) {
			case rw::rqw::ImageSaveFormatRefactor::JPEG:  return "jpg";
			case rw::rqw::ImageSaveFormatRefactor::PNG:   return "png";
			case rw::rqw::ImageSaveFormatRefactor::BMP:   return "bmp";
			default: return "jpg";
			}
		}
	}
}
