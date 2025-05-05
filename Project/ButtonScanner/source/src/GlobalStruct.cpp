#include "stdafx.h"

#include "GlobalStruct.h"
#include"hoec_CameraException.hpp"
#include"rqw_CameraObjectThread.hpp"

void GlobalStructData::setCameraExposureTime(int cameraIndex, size_t exposureTime)
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
	case 3:
		if (camera3) {
			camera3->setExposureTime(exposureTime);
			if (exposureTime < 200) {
				camera3->setGain(0);
			}
			else {
				camera3->setGain(5);
			}
		}        break;
	case 4:
		if (camera4) {
			camera4->setExposureTime(exposureTime);
			if (exposureTime < 200) {
				camera4->setGain(0);
			}
			else {
				camera4->setGain(5);
			}
		}        break;
	default:
		break;
	}
}

void GlobalStructData::buildImageProcessingModule(size_t num)
{
	imageProcessingModule1 = std::make_unique<ImageProcessingModule>(num, this);
	imageProcessingModule1->modelEnginePath = enginePath;
	imageProcessingModule1->modelNamePath = namePath;
	imageProcessingModule1->modelOnnxOOPath = onnxEngineOOPath;
	imageProcessingModule1->modelOnnxSOPath = onnxEngineSOPath;
	imageProcessingModule1->index = 1;
	imageProcessingModule1->BuildModule();

	imageProcessingModule2 = std::make_unique<ImageProcessingModule>(num, this);
	imageProcessingModule2->modelEnginePath = enginePath;
	imageProcessingModule2->modelNamePath = namePath;
	imageProcessingModule2->modelOnnxOOPath = onnxEngineOOPath;
	imageProcessingModule2->modelOnnxSOPath = onnxEngineSOPath;
	imageProcessingModule2->index = 2;
	imageProcessingModule2->BuildModule();

	imageProcessingModule3 = std::make_unique<ImageProcessingModule>(num, this);
	imageProcessingModule3->modelEnginePath = enginePath;
	imageProcessingModule3->modelNamePath = namePath;
	imageProcessingModule3->modelOnnxOOPath = onnxEngineOOPath;
	imageProcessingModule3->modelOnnxSOPath = onnxEngineSOPath;
	imageProcessingModule3->index = 3;
	imageProcessingModule3->BuildModule();
	auto processers3 = imageProcessingModule3->getProcessors();

	imageProcessingModule4 = std::make_unique<ImageProcessingModule>(num, this);
	imageProcessingModule4->modelEnginePath = enginePath;
	imageProcessingModule4->modelNamePath = namePath;
	imageProcessingModule4->modelOnnxOOPath = onnxEngineOOPath;
	imageProcessingModule4->modelOnnxSOPath = onnxEngineSOPath;
	imageProcessingModule4->index = 4;
	imageProcessingModule4->BuildModule();
	auto processers4 = imageProcessingModule4->getProcessors();
}

void GlobalStructData::buildConfigManager(rw::oso::StorageType type)
{
	storeContext = std::make_unique<rw::oso::StorageContext>(type);
}

void GlobalStructData::ReadConfig()
{
	ReadMainWindowConfig();
	ReadDlgProduceLineSetConfig();
	ReadDlgProductSetConfig();
	ReadDlgExposureTimeSetConfig();
	ReadDlgHideScoreSetConfig();
}

void GlobalStructData::ReadMainWindowConfig()
{
	auto loadMainWindowConfig = storeContext->load(mainWindowFilePath.toStdString());
	if (loadMainWindowConfig) {
		mainWindowConfig = *loadMainWindowConfig;
		isTakePictures = mainWindowConfig.isTakePictures;
		statisticalInfo.produceCount = mainWindowConfig.totalProduction;
		statisticalInfo.wasteCount = mainWindowConfig.totalWaste;
		statisticalInfo.productionYield = mainWindowConfig.passRate;
		statisticalInfo.removeRate = mainWindowConfig.scrappingRate;
		isOpenDefect = mainWindowConfig.isDefect;
	}
	else {
		LOG()  "Load main window config failed.";
	}
}

void GlobalStructData::ReadDlgProduceLineSetConfig()
{
	auto loadDlgProduceLineSetConfig = storeContext->load(dlgProduceLineSetFilePath.toStdString());
	if (loadDlgProduceLineSetConfig) {
		dlgProduceLineSetConfig = *loadDlgProduceLineSetConfig;
	}
	else {
		LOG()  "Load main window config failed.";
	}
}

void GlobalStructData::ReadDlgProductSetConfig()
{
	auto loadDlgProductSetConfig = storeContext->load(dlgProductSetFilePath.toStdString());
	if (loadDlgProductSetConfig) {
		dlgProductSetConfig = *loadDlgProductSetConfig;
	}
	else {
		LOG()  "Load main window config failed.";
	}
}

void GlobalStructData::ReadDlgExposureTimeSetConfig()
{
	auto loadDlgExposureTimeSetConfig = storeContext->load(dlgExposureTimeSetFilePath.toStdString());
	if (loadDlgExposureTimeSetConfig) {
		dlgExposureTimeSetConfig = *loadDlgExposureTimeSetConfig;
	}
	else {
		LOG()  "Load main window config failed.";
	}
}

void GlobalStructData::ReadDlgHideScoreSetConfig()
{
	auto loadDlgHideScoreSetConfig = storeContext->load(dlgHideScoreSetPath.toStdString());
	if (loadDlgHideScoreSetConfig) {
		dlgHideScoreSetConfig = *loadDlgHideScoreSetConfig;
	}
	else {
		LOG()  "Load main window config failed.";
	}
}

void GlobalStructData::saveConfig()
{
	saveMainWindowConfig();
	saveDlgProduceLineSetConfig();
	saveDlgProductSetConfig();
	saveDlgExposureTimeSetConfig();
	saveDlgHideScoreSetConfig();
}

void GlobalStructData::saveMainWindowConfig()
{
	storeContext->save(mainWindowConfig, mainWindowFilePath.toStdString());
}

void GlobalStructData::saveDlgProduceLineSetConfig()
{
	storeContext->save(dlgProduceLineSetConfig, dlgProduceLineSetFilePath.toStdString());
}

void GlobalStructData::saveDlgProductSetConfig() {
	storeContext->save(dlgProductSetConfig, dlgProductSetFilePath.toStdString());
}

void GlobalStructData::saveDlgExposureTimeSetConfig()
{
	storeContext->save(dlgExposureTimeSetConfig, dlgExposureTimeSetFilePath.toStdString());
}

void GlobalStructData::saveDlgHideScoreSetConfig()
{
	storeContext->save(dlgHideScoreSetConfig, dlgHideScoreSetPath.toStdString());
}

bool GlobalStructData::isTargetCamera(const QString& cameraIndex, const QString& targetName)
{
	QRegularExpression regex(R"((\d+)\.(\d+)\.(\d+)\.(\d+))");
	QRegularExpressionMatch match = regex.match(targetName);

	if (match.hasMatch()) {
		auto matchString = match.captured(3);

		return cameraIndex == matchString;
	}

	return false;
}

rw::rqw::CameraMetaData GlobalStructData::cameraMetaDataCheck(const QString& cameraIndex, const QVector<rw::rqw::CameraMetaData>& cameraInfo)
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

void GlobalStructData::buildCamera()
{
	buildCamera1();
	buildCamera2();
	buildCamera3();
	buildCamera4();
}

bool GlobalStructData::buildCamera1()
{
	auto cameraList = rw::rqw::CheckCameraList();

	auto cameraMetaData1 = cameraMetaDataCheck(cameraIp1, cameraList);

	if (cameraMetaData1.ip != "0") {
		try
		{
			camera1 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera1->motionRedix = 2;
			camera1->initCamera(cameraMetaData1, rw::rqw::CameraObjectTrigger::Hardware, 2);
			camera1->cameraIndex = 1;
			camera1->setHeartbeatTime(5000);
			setCameraExposureTime(1, dlgExposureTimeSetConfig.expousureTime);
			QObject::connect(camera1.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				imageProcessingModule1.get(), &ImageProcessingModule::onFrameCaptured, Qt::DirectConnection);
			return true;
		}
		catch (const std::exception&)
		{
			return false;
			LOG()  "Camera 1 initialization failed.";
		}
	}
	return false;
}

bool GlobalStructData::buildCamera2()
{
	auto cameraList = rw::rqw::CheckCameraList();
	auto cameraMetaData2 = cameraMetaDataCheck(cameraIp2, cameraList);
	if (cameraMetaData2.ip != "0") {
		try
		{
			camera2 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera2->motionRedix = 4;
			camera2->initCamera(cameraMetaData2, rw::rqw::CameraObjectTrigger::Hardware, 4);
			camera2->cameraIndex = 2;
			camera2->setHeartbeatTime(5000);
			setCameraExposureTime(2, dlgExposureTimeSetConfig.expousureTime);
			QObject::connect(camera2.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				imageProcessingModule2.get(), &ImageProcessingModule::onFrameCaptured, Qt::DirectConnection);
			return true;
		}
		catch (const std::exception&)
		{
			return false;
			LOG()  "Camera 2 initialization failed.";
		}
	}
	return false;
}

bool GlobalStructData::buildCamera3()
{
	auto cameraList = rw::rqw::CheckCameraList();
	auto cameraMetaData3 = cameraMetaDataCheck(cameraIp3, cameraList);
	if (cameraMetaData3.ip != "0") {
		try
		{
			camera3 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera3->motionRedix = 6;
			camera3->initCamera(cameraMetaData3, rw::rqw::CameraObjectTrigger::Hardware, 6);
			camera3->cameraIndex = 3;
			camera3->setHeartbeatTime(5000);
			setCameraExposureTime(3, dlgExposureTimeSetConfig.expousureTime);
			QObject::connect(camera3.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				imageProcessingModule3.get(), &ImageProcessingModule::onFrameCaptured, Qt::DirectConnection);
			return true;
		}
		catch (const std::exception&)
		{
			return false;
			LOG()  "Camera 3 initialization failed.";
		}
	}
	return false;
}

bool GlobalStructData::buildCamera4()
{
	auto cameraList = rw::rqw::CheckCameraList();
	auto cameraMetaData4 = cameraMetaDataCheck(cameraIp4, cameraList);
	if (cameraMetaData4.ip != "0") {
		try
		{
			camera4 = std::make_unique<rw::rqw::CameraPassiveThread>(this);
			camera4->motionRedix = 8;
			camera4->initCamera(cameraMetaData4, rw::rqw::CameraObjectTrigger::Hardware, 8);
			camera4->cameraIndex = 4;
			camera4->setHeartbeatTime(5000);
			setCameraExposureTime(4, dlgExposureTimeSetConfig.expousureTime);
			QObject::connect(camera4.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
				imageProcessingModule4.get(), &ImageProcessingModule::onFrameCaptured, Qt::DirectConnection);
			return true;
		}
		catch (const std::exception&)
		{
			return false;
			LOG()  "Camera 4 initialization failed.";
		}
	}
	return false;
}

void GlobalStructData::startMonitor()
{
	if (camera1)
	{
		try
		{
			camera1->startMonitor();
		}
		catch (rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 1 startMonitor failed: " << e.what();
		}
	}
	if (camera2)
	{
		try
		{
			camera2->startMonitor();
		}
		catch (rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 2 startMonitor failed: " << e.what();
		}
	}
	if (camera3)
	{
		try
		{
			camera3->startMonitor();
		}
		catch (const rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 3 startMonitor failed: " << e.what();
		}
	}
	if (camera4)
	{
		try
		{
			camera4->startMonitor();
		}
		catch (const rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 4 startMonitor failed: " << e.what();
		}
	}
}

void GlobalStructData::destroyCamera()
{
	destroyCamera1();
	destroyCamera2();
	destroyCamera3();
	destroyCamera4();
}

void GlobalStructData::destroyCamera1()
{
	QObject::disconnect(camera1.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
		imageProcessingModule1.get(), &ImageProcessingModule::onFrameCaptured);
	camera1.reset();
}

void GlobalStructData::destroyCamera2()
{
	QObject::disconnect(camera2.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
		imageProcessingModule2.get(), &ImageProcessingModule::onFrameCaptured);
	camera2.reset();
}

void GlobalStructData::destroyCamera3()
{
	QObject::disconnect(camera3.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
		imageProcessingModule3.get(), &ImageProcessingModule::onFrameCaptured);
	camera3.reset();
}

void GlobalStructData::destroyCamera4()
{
	QObject::disconnect(camera4.get(), &rw::rqw::CameraPassiveThread::frameCaptured,
		imageProcessingModule4.get(), &ImageProcessingModule::onFrameCaptured);
	camera4.reset();
}

void GlobalStructData::destroyImageProcessingModule()
{
	imageProcessingModule1.reset();
	imageProcessingModule2.reset();
	imageProcessingModule3.reset();
	imageProcessingModule4.reset();
}

void GlobalStructData::buildImageSaveEngine()
{
	imageSaveEngine = std::make_unique<rw::rqw::ImageSaveEngine>(this, 4);
}

void GlobalStructData::destroyImageSaveEngine()
{
	imageSaveEngine->stop();
	imageSaveEngine.reset();
}

GlobalStructData::GlobalStructData()
{
}

void GlobalStructData::setUpLight(bool state)
{
	mainWindowConfig.upLight = state;
	emit updateLightState(0, state);
}

void GlobalStructData::setDownLight(bool state)
{
	mainWindowConfig.downLight = state;
	emit updateLightState(1, state);
}

void GlobalStructData::setSideLight(bool state)
{
	mainWindowConfig.sideLight = state;
	emit updateLightState(2, state);
}

void GlobalStructData::onBuildCamera1()
{
	buildCamera1();
}

void GlobalStructData::onBuildCamera2()
{
	buildCamera2();
}

void GlobalStructData::onBuildCamera3()
{
	buildCamera3();
}

void GlobalStructData::onBuildCamera4()
{
	buildCamera4();
}

void GlobalStructData::onDestroyCamera1()
{
	destroyCamera1();
}

void GlobalStructData::onDestroyCamera2()
{
	destroyCamera2();
}

void GlobalStructData::onDestroyCamera3()
{
	destroyCamera3();
}

void GlobalStructData::onDestroyCamera4()
{
	destroyCamera4();
}

void GlobalStructData::onStartMonitor1()
{
	if (camera1)
	{
		try
		{
			camera1->startMonitor();
		}
		catch (rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 1 startMonitor failed: " << e.what();
		}
	}
}

void GlobalStructData::onStartMonitor2()
{
	if (camera2)
	{
		try
		{
			camera2->startMonitor();
		}
		catch (rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 2 startMonitor failed: " << e.what();
		}
	}
}

void GlobalStructData::onStartMonitor3()
{
	if (camera3)
	{
		try
		{
			camera3->startMonitor();
		}
		catch (rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 3 startMonitor failed: " << e.what();
		}
	}
}

void GlobalStructData::onStartMonitor4()
{
	if (camera4)
	{
		try
		{
			camera4->startMonitor();
		}
		catch (rw::hoec::CameraMonitorError& e)
		{
			LOG()  "Camera 4 startMonitor failed: " << e.what();
		}
	}
}

void GlobalStructThread::buildDetachThread()
{
	auto& instance = GlobalStructData::getInstance();
	statisticalInfoComputingThread = std::make_unique<StatisticalInfoComputingThread>(this);
	statisticalInfoComputingThread->startThread();

	monitorCameraAndCardStateThread = std::make_unique<MonitorCameraAndCardStateThread>(this);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::buildCamera1,
		&instance, &GlobalStructData::onBuildCamera1, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::buildCamera2,
		&instance, &GlobalStructData::onBuildCamera2, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::buildCamera3,
		&instance, &GlobalStructData::onBuildCamera3, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::buildCamera4,
		&instance, &GlobalStructData::onBuildCamera4, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::destroyCamera1,
		&instance, &GlobalStructData::onDestroyCamera1, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::destroyCamera2,
		&instance, &GlobalStructData::onDestroyCamera2, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::destroyCamera3,
		&instance, &GlobalStructData::onDestroyCamera3, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::destroyCamera4,
		&instance, &GlobalStructData::onDestroyCamera4, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::startMonitor1,
		&instance, &GlobalStructData::onStartMonitor1, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::startMonitor2,
		&instance, &GlobalStructData::onStartMonitor2, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::startMonitor3,
		&instance, &GlobalStructData::onStartMonitor3, Qt::QueuedConnection);
	QObject::connect(monitorCameraAndCardStateThread.get(), &MonitorCameraAndCardStateThread::startMonitor4,
		&instance, &GlobalStructData::onStartMonitor4, Qt::QueuedConnection);

	monitorCameraAndCardStateThread->startThread();

	aiTrainModule = std::make_unique<AiTrainModule>(this);
}

void GlobalStructThread::destroyDetachThread()
{
	statisticalInfoComputingThread->stopThread();
	monitorCameraAndCardStateThread->stopThread();

	statisticalInfoComputingThread->wait();
	statisticalInfoComputingThread->wait();

	monitorCameraAndCardStateThread.reset();
	statisticalInfoComputingThread.reset();
	aiTrainModule.reset();
}

GlobalStructThread::GlobalStructThread()
{
}