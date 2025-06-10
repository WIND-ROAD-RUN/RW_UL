#include "GlobalStruct.hpp"

#include <qregularexpression.h>

#include "rqw_CameraObjectCore.hpp"

void GlobalStructDataSmartCroppingOfBags::onCameraReject(size_t index)
{

}

GlobalStructDataSmartCroppingOfBags::GlobalStructDataSmartCroppingOfBags()
{

}

GlobalStructDataSmartCroppingOfBags::~GlobalStructDataSmartCroppingOfBags()
{

}

void GlobalStructDataSmartCroppingOfBags::buildConfigManager(rw::oso::StorageType type)
{
	storeContext = std::make_unique<rw::oso::StorageContext>(type);
}

void GlobalStructDataSmartCroppingOfBags::buildImageSaveEngine()
{
	imageSaveEngine = std::make_unique<rw::rqw::ImageSaveEngine>(this, 2);
}

void GlobalStructDataSmartCroppingOfBags::destroyImageSaveEngine()
{
	imageSaveEngine->stop();
	imageSaveEngine.reset();
}

void GlobalStructDataSmartCroppingOfBags::saveGeneralConfig()
{
	std::string generalConfigPath = globalPath.generalConfigPath.toStdString();
	storeContext->save(generalConfig, generalConfigPath);
}

void GlobalStructDataSmartCroppingOfBags::saveDlgProductSetConfig()
{
	std::string setConfigPath = globalPath.setConfigPath.toStdString();
	storeContext->save(setConfig, setConfigPath);
}

void GlobalStructDataSmartCroppingOfBags::buildCamera()
{
	buildCamera1();
	buildCamera2();
}

bool GlobalStructDataSmartCroppingOfBags::buildCamera1()
{
	auto cameraList = rw::rqw::CheckCameraList();

	auto cameraMetaData1 = cameraMetaDataCheck(cameraIp1, cameraList);

	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	// 剔废持续时间
	//long DurationTime = setConfig.tiFeiChiXuShiJian1 * 1000;

	if (cameraMetaData1.ip != "0")
	{
		try
		{
			camera1 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera1->initCamera(cameraMetaData1, rw::rqw::CameraObjectTrigger::Software);
			camera1->cameraIndex = 1;
			camera1->setHeartbeatTime(5000);
			//setCameraExposureTime(1, dlgExposureTimeSetConfig.exposureTime);
			camera1->startMonitor();
			// 设置剔废IO输出
			//auto config = rw::rqw::OutTriggerConfig({ 2,8,5,DurationTime,0,0,true });
			//camera1->setOutTriggerConfig(config);
			return true;
		}
		catch (const std::exception&)
		{
			return false;
			//LOG()  "Camera 1 initialization failed.";
		}
	}
	return false;
}

bool GlobalStructDataSmartCroppingOfBags::buildCamera2()
{
	auto cameraList = rw::rqw::CheckCameraList();

	auto cameraMetaData2 = cameraMetaDataCheck(cameraIp2, cameraList);

	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	// 剔废持续时间
	//long DurationTime = setConfig.tiFeiChiXuShiJian2 * 1000;

	if (cameraMetaData2.ip != "0")
	{
		try
		{
			camera2 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera2->initCamera(cameraMetaData2, rw::rqw::CameraObjectTrigger::Software);
			camera2->cameraIndex = 2;
			camera2->setHeartbeatTime(5000);
			//setCameraExposureTime(2, dlgExposureTimeSetConfig.exposureTime);
			// 设置剔废IO输出
			//auto config = rw::rqw::OutTriggerConfig({ 2,8,5,DurationTime,0,0,true });
			//camera2->setOutTriggerConfig(config);
			camera2->startMonitor();
			return true;
		}
		catch (const std::exception&)
		{
			return false;
			//LOG()  "Camera 2 initialization failed.";
		}
	}
	return false;
}

void GlobalStructDataSmartCroppingOfBags::destroyCamera()
{
	destroyCamera1();
	destroyCamera2();
}

void GlobalStructDataSmartCroppingOfBags::destroyCamera1()
{
	camera1.reset();
}

void GlobalStructDataSmartCroppingOfBags::destroyCamera2()
{
	camera2.reset();
}

bool GlobalStructDataSmartCroppingOfBags::isTargetCamera(const QString& cameraIndex, const QString& targetName)
{
	QRegularExpression regex(R"((\d+)\.(\d+)\.(\d+)\.(\d+))");
	QRegularExpressionMatch match = regex.match(targetName);

	if (match.hasMatch()) {
		auto matchString = match.captured(3);

		return cameraIndex == matchString;
	}

	return false;
}

rw::rqw::CameraMetaData GlobalStructDataSmartCroppingOfBags::cameraMetaDataCheck(const QString& cameraIndex,
                                                                                 const QVector<rw::rqw::CameraMetaData>& cameraInfo)
{
	for (const auto& cameraMetaData : cameraInfo) {
		if (isTargetCamera(cameraIndex, cameraMetaData.ip)) {
			return cameraMetaData;
		}
	}
	rw::rqw::CameraMetaData error;
	error.ip = "0";
	return error;
}

void GlobalStructDataSmartCroppingOfBags::setCameraExposureTime(int cameraIndex, size_t exposureTime)
{
	switch (cameraIndex) {
	case 1:
		if (camera1) {
			camera1->setExposureTime(exposureTime);
			if (exposureTime < 200) {
				camera1->setGain(0);
			}
			else {
				camera1->setGain(5);
			}
		}
		break;
	case 2:
		if (camera2) {
			camera2->setExposureTime(exposureTime);
			if (exposureTime < 200) {
				camera2->setGain(0);
			}
			else {
				camera2->setGain(5);
			}
		}
		break;
	default:
		break;
	}
}
