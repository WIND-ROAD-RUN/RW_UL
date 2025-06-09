#include"GlobalStruct.hpp"

#include <qregularexpression.h>

#include "hoec_Camera.hpp"
#include "rqw_CameraObjectCore.hpp"
#include "Utilty.hpp"


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


	//using namespace std::chrono;
	//// 获取当前时间点
	//auto now = system_clock::now();
	//auto now_time = Time(now);
	//// 插入当前时间点
	//priorityQueue1->insert(now_time, now_time);
	//// 插入当前时间点加1秒
	//auto t1 = Time(now + seconds(1));
	//priorityQueue1->insert(t1, t1);
	//// 插入当前时间点加5秒
	//auto t2 = Time(now + seconds(5));
	//priorityQueue1->insert(t2, t2);
	//// 插入当前时间点减10秒
	//auto t3 = Time(now - seconds(10));
	//priorityQueue1->insert(t3, t3);

	////这里是取出逻辑
	//Time preTime;
	//try
	//{
	//	preTime=priorityQueue1->peek();
	//}
	//catch (const std::runtime_error&)
	//{
	//	return;
	//}

	//if (static_cast<double>(400)> 300) {
	//	// 这里可以添加一些逻辑处理
	//	priorityQueue1->top();
	//}

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

void GlobalStructDataZipper::buildImageSaveEngine()
{
	imageSaveEngine = std::make_unique<rw::rqw::ImageSaveEngine>(this, 2);
}

void GlobalStructDataZipper::destroyImageSaveEngine()
{
	imageSaveEngine->stop();
	imageSaveEngine.reset();
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

