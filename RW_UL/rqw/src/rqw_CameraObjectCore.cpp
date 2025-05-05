#include"rqw_CameraObjectCore.hpp"

#include"hoec_CameraFactory.hpp"
#include"hoec_Camera.hpp"

namespace rw
{
	namespace rqw
	{
		QVector<CameraMetaData> CheckCameraList()
		{
			auto stdCameraIpList = hoec::CameraFactory::checkAllCamera();
			QVector<CameraMetaData> cameraIpList;
			for (auto& cameraIp : stdCameraIpList)
			{
				CameraMetaData cameraMetaData;
				cameraMetaData.ip = QString::fromStdString(cameraIp.ip);
				cameraMetaData.provider = QString::fromStdString(hoec::to_string(cameraIp.provider));
				cameraIpList.push_back(cameraMetaData);
			}

			return cameraIpList;
		}
	} // namespace rqw
} // namespace rw