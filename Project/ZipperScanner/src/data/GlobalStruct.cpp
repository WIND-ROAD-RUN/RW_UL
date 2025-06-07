#include"GlobalStruct.hpp"

#include <qregularexpression.h>

#include "hoec_Camera.hpp"
#include "rqw_CameraObjectCore.hpp"
#include "Utilty.hpp"


GlobalStructDataZipper::GlobalStructDataZipper()
{
}

bool GlobalStructDataZipper::isTargetCamera(const QString& cameraIndex, const QString& targetName)
{
	QRegularExpression regex(R"((\d+)\.(\d+)\.(\d+)\.(\d+))");
	QRegularExpressionMatch match = regex.match(targetName);

	if (match.hasMatch()) {
		auto matchString = match.captured(3);

		return cameraIndex == matchString;
	}

	return false;
}

void GlobalStructDataZipper::setCameraExposureTime(int cameraIndex, size_t exposureTime)
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

void GlobalStructDataZipper::buildConfigManager(rw::oso::StorageType type)
{
	storeContext = std::make_unique<rw::oso::StorageContext>(type);
}

void GlobalStructDataZipper::buildImageProcessorModules(const QString& path)
{
	modelCamera1 = std::make_unique<ImageProcessingModuleZipper>(2);
	modelCamera2 = std::make_unique<ImageProcessingModuleZipper>(2);

	modelCamera1->modelEnginePath = path;
	modelCamera2->modelEnginePath = path;

	modelCamera1->BuildModule();
	modelCamera2->BuildModule();

	modelCamera1->index = 1;
	modelCamera2->index = 2;
}

void GlobalStructDataZipper::saveDlgProductSetConfig()
{
	std::string setConfigPath = globalPath.setConfigPath.toStdString();
	storeContext->save(setConfig, setConfigPath);
}

void GlobalStructDataZipper::saveDlgProductScoreConfig()
{
	std::string scoreConfigPath = globalPath.scoreConfigPath.toStdString();
	storeContext->save(scoreConfig, scoreConfigPath);
}

void GlobalStructDataZipper::buildCamera()
{
	buildCamera1();
	buildCamera2();
}

bool GlobalStructDataZipper::buildCamera1()
{
	auto cameraList = rw::rqw::CheckCameraList();

	auto cameraMetaData1 = cameraMetaDataCheck(cameraIp1, cameraList);

	if (cameraMetaData1.ip != "0")
	{
		try
		{
			camera1 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera1->initCamera(cameraMetaData1, rw::rqw::CameraObjectTrigger::Software);
			camera1->cameraIndex = 1;
			camera1->setHeartbeatTime(5000);
			setCameraExposureTime(1, dlgExposureTimeSetConfig.exposureTime);
			camera1->startMonitor();
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

bool GlobalStructDataZipper::buildCamera2()
{
	auto cameraList = rw::rqw::CheckCameraList();

	auto cameraMetaData2 = cameraMetaDataCheck(cameraIp2, cameraList);

	if (cameraMetaData2.ip != "0")
	{
		try
		{
			camera2 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera2->initCamera(cameraMetaData2, rw::rqw::CameraObjectTrigger::Software);
			camera2->cameraIndex = 2;
			camera2->setHeartbeatTime(5000);
			setCameraExposureTime(2, dlgExposureTimeSetConfig.exposureTime);
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

void GlobalStructDataZipper::destroyCamera()
{
	destroyCamera1();
	destroyCamera2();
}

void GlobalStructDataZipper::destroyCamera1()
{
	camera1.reset();
}

void GlobalStructDataZipper::destroyCamera2()
{
	camera2.reset();
}

rw::rqw::CameraMetaData GlobalStructDataZipper::cameraMetaDataCheck(const QString& cameraIndex, const QVector<rw::rqw::CameraMetaData>& cameraInfo)
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

