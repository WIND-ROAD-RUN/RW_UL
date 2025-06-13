#include"GlobalStruct.hpp"

#include <qregularexpression.h>

#include "hoec_Camera.hpp"
#include "rqw_CameraObjectCore.hpp"
#include "Utilty.hpp"
#include "DetachDefectThread.h"
#include "rqw_CameraObjectThreadZMotion.hpp"


void GlobalStructDataZipper::build_PriorityQueue()
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

void GlobalStructDataZipper::destroy_PriorityQueue()
{
	priorityQueue1.reset();
	priorityQueue2.reset();
}

void GlobalStructDataZipper::build_DetachDefectThreadZipper()
{
	detachDefectThreadZipper = new DetachDefectThreadZipper(this);

	// 连接剔废功能
	QObject::connect(detachDefectThreadZipper, &DetachDefectThreadZipper::findIsBad
		,this, &GlobalStructDataZipper::onCameraReject);
}

void GlobalStructDataZipper::destroy_DetachDefectThreadZipper()
{
	if (detachDefectThreadZipper)
	{
		delete detachDefectThreadZipper;
	}
}

void GlobalStructDataZipper::build_CameraAndCardStateThreadZipper()
{
	cameraAndCardStateThreadZipper = new CameraAndCardStateThreadZipper(this);
	// 更新UI界面
	QObject::connect(cameraAndCardStateThreadZipper, &CameraAndCardStateThreadZipper::updateCameraLabelState,
		this, &GlobalStructDataZipper::emit_updateUiLabels, Qt::QueuedConnection);
	// 相机重连
	QObject::connect(cameraAndCardStateThreadZipper, &CameraAndCardStateThreadZipper::buildCamera1,
		this, &GlobalStructDataZipper::rebuild_Camera1, Qt::QueuedConnection);
	QObject::connect(cameraAndCardStateThreadZipper, &CameraAndCardStateThreadZipper::buildCamera2,
		this, &GlobalStructDataZipper::rebuild_Camera2, Qt::QueuedConnection);
	// 相机销毁
	QObject::connect(cameraAndCardStateThreadZipper, &CameraAndCardStateThreadZipper::destroyCamera1,
		this, &GlobalStructDataZipper::destroy_Camera1, Qt::QueuedConnection);
	QObject::connect(cameraAndCardStateThreadZipper, &CameraAndCardStateThreadZipper::destroyCamera2,
		this, &GlobalStructDataZipper::destroy_Camera2, Qt::QueuedConnection);
}

void GlobalStructDataZipper::rebuild_Camera1()
{
	auto cameraList = rw::rqw::CheckCameraList();

	auto cameraMetaData1 = cameraMetaDataCheck(cameraIp1, cameraList);

	// 剔废持续时间
	long DurationTime = setConfig.tiFeiChiXuShiJian1 * 1000;

	if (cameraMetaData1.ip != "0")
	{
		try
		{
			camera1 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera1->initCamera(cameraMetaData1, rw::rqw::CameraObjectTrigger::Hardware);
			camera1->cameraIndex = 1;
			camera1->setFrameRate(50);
			camera1->setHeartbeatTime(5000);
			if (generalConfig.qiangGuang == true)
			{
				// 强光照明
				setLightLevel(LightLevel::StrongLight);
			}
			else if (generalConfig.zhongGuang == true)
			{
				// 中光照明
				setLightLevel(LightLevel::MediumLight);
			}
			else if (generalConfig.ruoGuang == true)
			{
				// 弱光照明
				setLightLevel(LightLevel::WeakLight);
			}
			camera1->startMonitor();
			// 设置剔废IO输出
			auto config = rw::rqw::OutTriggerConfig({ 2,8,5,DurationTime,0,0,true });
			camera1->setOutTriggerConfig(config);
			QObject::connect(camera1.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				modelCamera1.get(), &ImageProcessingModuleZipper::onFrameCaptured, Qt::DirectConnection);
		}
		catch (const std::exception&)
		{
			//LOG()  "Camera 1 initialization failed.";
		}
	}
}

void GlobalStructDataZipper::rebuild_Camera2()
{
	auto cameraList = rw::rqw::CheckCameraList();

	auto cameraMetaData2 = cameraMetaDataCheck(cameraIp2, cameraList);

	// 剔废持续时间
	long DurationTime = setConfig.tiFeiChiXuShiJian2 * 1000;

	if (cameraMetaData2.ip != "0")
	{
		try
		{
			camera2 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera2->initCamera(cameraMetaData2, rw::rqw::CameraObjectTrigger::Hardware);
			camera2->cameraIndex = 2;
			camera2->setFrameRate(50);
			camera2->setHeartbeatTime(5000);
			if (generalConfig.qiangGuang == true)
			{
				// 强光照明
				setLightLevel(LightLevel::StrongLight);
			}
			else if (generalConfig.zhongGuang == true)
			{
				// 中光照明
				setLightLevel(LightLevel::MediumLight);
			}
			else if (generalConfig.ruoGuang == true)
			{
				// 弱光照明
				setLightLevel(LightLevel::WeakLight);
			}
			// 设置剔废IO输出
			auto config = rw::rqw::OutTriggerConfig({ 2,8,5,DurationTime,0,0,true });
			camera2->setOutTriggerConfig(config);
			camera2->startMonitor();
			QObject::connect(camera2.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				modelCamera2.get(), &ImageProcessingModuleZipper::onFrameCaptured, Qt::DirectConnection);
		}
		catch (const std::exception&)
		{
			//LOG()  "Camera 2 initialization failed.";
		}
	}
}

void GlobalStructDataZipper::destroy_Camera1()
{
	destroyCamera1();
}

void GlobalStructDataZipper::destroy_Camera2()
{
	destroyCamera2();
}

void GlobalStructDataZipper::onCameraReject(size_t index)
{
	auto& globalStruct = GlobalStructDataZipper::getInstance();
	if (globalStruct.runningState == RunningState::OpenRemoveFunc)
	{
		if (index == 1)
		{
			camera1->outTrigger();
		}
		else if (index == 2)
		{
			camera2->outTrigger();
		}
	}
}

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

void GlobalStructDataZipper::setLightLevel(const LightLevel& level)
{
	switch (level)
	{
	case LightLevel::StrongLight :
		if (camera1) {
			camera1->setExposureTime(setConfig.qiangBaoGuang);
			camera1->setGain(setConfig.qiangZengYi);
		}
		if (camera2) {
			camera2->setExposureTime(setConfig.qiangBaoGuang);
			camera2->setGain(setConfig.qiangZengYi);
		}
		break;
	case LightLevel::MediumLight:
		if (camera1) {
			camera1->setExposureTime(setConfig.zhongBaoGuang);
			camera1->setGain(setConfig.zhongZengYi);
		}
		if (camera2) {
			camera2->setExposureTime(setConfig.zhongBaoGuang);
			camera2->setGain(setConfig.zhongZengYi);
		}
		break;
	case LightLevel::WeakLight:
		if (camera1) {
			camera1->setExposureTime(setConfig.ruoBaoGuang);
			camera1->setGain(setConfig.ruoZengYi);
		}
		if (camera2) {
			camera2->setExposureTime(setConfig.ruoBaoGuang);
			camera2->setGain(setConfig.ruoZengYi);
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

	modelCamera1->index = 1;
	modelCamera2->index = 2;

	modelCamera1->BuildModule();
	modelCamera2->BuildModule();

}

void GlobalStructDataZipper::destroyImageProcessingModule()
{
	modelCamera1.reset();
	modelCamera2.reset();
}

void GlobalStructDataZipper::buildImageSaveEngine()
{
	imageSaveEngine = std::make_unique<rw::rqw::ImageSaveEngine>(this, 2);
}

void GlobalStructDataZipper::destroyImageSaveEngine()
{
	imageSaveEngine->stop();
	imageSaveEngine.reset();
}

void GlobalStructDataZipper::saveGeneralConfig()
{
	std::string generalConfigPath = globalPath.generalConfigPath.toStdString();
	storeContext->save(generalConfig, generalConfigPath);
}

void GlobalStructDataZipper::saveDlgProductSetConfig()
{
	// 调试模式默认为不开启
	setConfig.debugMode = false;
	std::string setConfigPath = globalPath.setConfigPath.toStdString();
	storeContext->save(setConfig, setConfigPath);
}

void GlobalStructDataZipper::saveDlgProductScoreConfig()
{
	std::string scoreConfigPath = globalPath.scoreConfigPath.toStdString();
	storeContext->save(scoreConfig, scoreConfigPath);
}

void GlobalStructDataZipper::saveDlgExposureTimeSetConfig()
{
	storeContext->save(dlgExposureTimeSetConfig, globalPath.dlgExposureTimeSetFilePath.toStdString());
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

	// 剔废持续时间
	long DurationTime = setConfig.tiFeiChiXuShiJian1 * 1000;

	if (cameraMetaData1.ip != "0")
	{
		try
		{
			camera1 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera1->initCamera(cameraMetaData1, rw::rqw::CameraObjectTrigger::Hardware);
			camera1->cameraIndex = 1;
			camera1->setFrameRate(50);
			camera1->setHeartbeatTime(5000);
			if (generalConfig.qiangGuang == true)
			{
				// 强光照明
				setLightLevel(LightLevel::StrongLight);
			}
			else if (generalConfig.zhongGuang == true)
			{
				// 中光照明
				setLightLevel(LightLevel::MediumLight);
			}
			else if (generalConfig.ruoGuang == true)
			{
				// 弱光照明
				setLightLevel(LightLevel::WeakLight);
			}
			camera1->startMonitor();
			// 设置剔废IO输出
			auto config = rw::rqw::OutTriggerConfig({2,8,5,DurationTime,0,0,true});
			camera1->setOutTriggerConfig(config);
			QObject::connect(camera1.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				modelCamera1.get(), &ImageProcessingModuleZipper::onFrameCaptured, Qt::DirectConnection);
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

	// 剔废持续时间
	long DurationTime = setConfig.tiFeiChiXuShiJian2 * 1000;

	if (cameraMetaData2.ip != "0")
	{
		try
		{
			camera2 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera2->initCamera(cameraMetaData2, rw::rqw::CameraObjectTrigger::Hardware);
			camera2->cameraIndex = 2;
			camera2->setFrameRate(50);
			camera2->setHeartbeatTime(5000);
			if (generalConfig.qiangGuang == true)
			{
				// 强光照明
				setLightLevel(LightLevel::StrongLight);
			}
			else if (generalConfig.zhongGuang == true)
			{
				// 中光照明
				setLightLevel(LightLevel::MediumLight);
			}
			else if (generalConfig.ruoGuang == true)
			{
				// 弱光照明
				setLightLevel(LightLevel::WeakLight);
			}
			// 设置剔废IO输出
			auto config = rw::rqw::OutTriggerConfig({ 2,8,5,DurationTime,0,0,true });
			camera2->setOutTriggerConfig(config);
			camera2->startMonitor();
			QObject::connect(camera2.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				modelCamera2.get(), &ImageProcessingModuleZipper::onFrameCaptured, Qt::DirectConnection);
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
	QObject::disconnect(camera1.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
		modelCamera1.get(), &ImageProcessingModuleZipper::onFrameCaptured);
	camera1.reset();
}

void GlobalStructDataZipper::destroyCamera2()
{
	QObject::disconnect(camera2.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
		modelCamera2.get(), &ImageProcessingModuleZipper::onFrameCaptured);
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

