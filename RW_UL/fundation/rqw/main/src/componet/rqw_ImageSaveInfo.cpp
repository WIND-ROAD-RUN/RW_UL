#include"rqw_ImageSaveInfo.hpp"


namespace rw
{
	namespace rqw
	{
		QString imageFormatToString(rw::rqw::ImageSaveFormatV1 format)
		{
			switch (format) {
			case rw::rqw::ImageSaveFormatV1::JPEG:  return "jpg";
			case rw::rqw::ImageSaveFormatV1::PNG:   return "png";
			case rw::rqw::ImageSaveFormatV1::BMP:   return "bmp";
			default: return "jpg";
			}
		}
	}
}
