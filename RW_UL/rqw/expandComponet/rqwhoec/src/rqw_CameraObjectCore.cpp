#include"rqw_CameraObjectCore.hpp"

#include"hoec_CameraFactory_v1.hpp"
#include"hoec_Camera_v1.hpp"

namespace rw
{
	namespace rqw
	{
		QVector<CameraMetaData> CheckCameraList()
		{
			auto stdCameraIpList = hoec_v1::CameraFactory::checkAllCamera();
			QVector<CameraMetaData> cameraIpList;
			for (auto& cameraIp : stdCameraIpList)
			{
				CameraMetaData cameraMetaData;
				cameraMetaData.ip = QString::fromStdString(cameraIp.ip);
				cameraMetaData.provider = QString::fromStdString(hoec_v1::to_string(cameraIp.provider));
				cameraIpList.push_back(cameraMetaData);
			}

			return cameraIpList;
		}
	} // namespace rqw
} // namespace rw