#include "GlobalStruct.hpp"

#include <qregularexpression.h>

#include "rqw_CameraObjectCore.hpp"
#include "DetachDefectThread.h"

bool GlobalStructDataSmartCroppingOfBags::build_motion()
{
	motion = std::make_unique<zwy::scc::Motion>();
	return motion->OpenBoard((char*)"192.168.0.11");
}

void GlobalStructDataSmartCroppingOfBags::destroy_motion()
{
	motion.reset();
}

void GlobalStructDataSmartCroppingOfBags::build_MonitorIOSmartCroppingOfBags()
{
	monitorIOSmartCroppingOfBags = std::make_unique<MonitorIOSmartCroppingOfBags>();
}

void GlobalStructDataSmartCroppingOfBags::destroy_MonitorIOSmartCroppingOfBags()
{
	monitorIOSmartCroppingOfBags.reset();
}

void GlobalStructDataSmartCroppingOfBags::build_PriorityQueue()
{
	auto compareNodeEqual = [](const Time& a, const Time& b) {
		return a == b;
		};
	auto compareNodePriority = [](const Time& a, const Time& b) {
		return a < b;
		};

	priorityQueue1 = std::make_unique<rw::dsl::ThreadSafeDHeap<Time, Time> >(compareNodeEqual, compareNodePriority);
	priorityQueue2 = std::make_unique<rw::dsl::ThreadSafeDHeap<Time, Time> >(compareNodeEqual, compareNodePriority);
}

void GlobalStructDataSmartCroppingOfBags::destroy_PriorityQueue()
{
	priorityQueue1.reset();
	priorityQueue2.reset();
}

void GlobalStructDataSmartCroppingOfBags::build_DetachDefectThreadSmartCroppingOfBags()
{
	detachDefectThreadSmartCroppingOfBags = new DetachDefectThreadSmartCroppingOfBags;

	// 连接剔废功能
	QObject::connect(detachDefectThreadSmartCroppingOfBags, &DetachDefectThreadSmartCroppingOfBags::findIsBad
		, this, &GlobalStructDataSmartCroppingOfBags::onCameraReject);
}

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

void GlobalStructDataSmartCroppingOfBags::buildImageProcessorModules(const QString& path)
{
	modelCamera1 = std::make_unique<ImageProcessingModuleSmartCroppingOfBags>(2);
	modelCamera2 = std::make_unique<ImageProcessingModuleSmartCroppingOfBags>(2);

	modelCamera1->modelEnginePath = path;
	modelCamera2->modelEnginePath = path;

	modelCamera1->index = 1;
	modelCamera2->index = 2;

	modelCamera1->BuildModule();
	modelCamera2->BuildModule();
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

void GlobalStructDataSmartCroppingOfBags::saveDlgProductScoreConfig()
{
	std::string scoreConfigPath = globalPath.scoreConfigPath.toStdString();
	storeContext->save(scoreConfig, scoreConfigPath);
}

void GlobalStructDataSmartCroppingOfBags::buildCamera()
{
	buildCamera1();
	buildCamera2();
}

bool GlobalStructDataSmartCroppingOfBags::buildCamera1()
{
	auto cameraList = rw::rqw::CheckCameraList(rw::rqw::CameraProvider::DS);

	auto cameraMetaData1 = cameraMetaDataCheck(cameraIp1, cameraList);

	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	// 剔废持续时间
	long DurationTime = setConfig.tifeishijian1 * 1000;

	if (cameraMetaData1.ip != "0")
	{
		try
		{
			camera1 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera1->initCamera(cameraMetaData1, rw::rqw::CameraObjectTrigger::Hardware);
			camera1->cameraIndex = 1;
			camera1->setFrameRate(50);
			camera1->setHeartbeatTime(5000);
			setCameraExposureTime(1, setConfig.xiangjibaoguang1);
			camera1->startMonitor();
			// 设置剔废IO输出
			//auto config = rw::rqw::OutTriggerConfig({ 2,8,5,DurationTime,0,0,true });
			//camera1->setOutTriggerConfig(config);
			QObject::connect(camera1.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				modelCamera1.get(), &ImageProcessingModuleSmartCroppingOfBags::onFrameCaptured, Qt::DirectConnection);
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
	auto cameraList = rw::rqw::CheckCameraList(rw::rqw::CameraProvider::DS);

	auto cameraMetaData2 = cameraMetaDataCheck(cameraIp2, cameraList);

	auto& setConfig = GlobalStructDataSmartCroppingOfBags::getInstance().setConfig;
	// 剔废持续时间
	//long DurationTime = setConfig.tiFeiChiXuShiJian2 * 1000;

	if (cameraMetaData2.ip != "0")
	{
		try
		{
			camera2 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera2->initCamera(cameraMetaData2, rw::rqw::CameraObjectTrigger::Hardware);
			camera2->cameraIndex = 2;
			camera1->setFrameRate(50);
			camera2->setHeartbeatTime(5000);
			setCameraExposureTime(2, setConfig.xiangjibaoguang1);
			// 设置剔废IO输出
			//auto config = rw::rqw::OutTriggerConfig({ 2,8,5,DurationTime,0,0,true });
			//camera2->setOutTriggerConfig(config);
			camera2->startMonitor();
			QObject::connect(camera2.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				modelCamera2.get(), &ImageProcessingModuleSmartCroppingOfBags::onFrameCaptured, Qt::DirectConnection);
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

void GlobalStructDataSmartCroppingOfBags::setCameraDebugMod()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();
	if (globalStruct.camera1) {
		globalStruct.camera1->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera1->setFrameRate(5);
	}

	if (globalStruct.camera2) {
		globalStruct.camera2->setTriggerMode(rw::rqw::CameraObjectTrigger::Software);
		globalStruct.camera2->setFrameRate(5);
	}
}

void GlobalStructDataSmartCroppingOfBags::setCameraDefectMod()
{
	auto& globalStruct = GlobalStructDataSmartCroppingOfBags::getInstance();

	if (globalStruct.camera1)
	{
		globalStruct.camera1->setTriggerMode(rw::rqw::CameraObjectTrigger::Hardware);
		globalStruct.camera1->setFrameRate(50);
	}
	if (globalStruct.camera2)
	{
		globalStruct.camera2->setTriggerMode(rw::rqw::CameraObjectTrigger::Hardware);
		globalStruct.camera2->setFrameRate(50);
	}
}
