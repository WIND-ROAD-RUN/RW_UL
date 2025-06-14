#pragma once

#include<memory>
#include<QString>
#include<QObject>

#include "GeneralConfig.hpp"
#include "SetConfig.hpp"
#include "DlgProductScore.h"
#include "Utilty.hpp"

#include<chrono>
#include <oso_StorageContext.hpp>

#include "rqw_CameraObjectThread.hpp"
#include "rqw_ImageSaveEngine.h"
#include <ScoreConfig.hpp>


#include "dsl_PriorityQueue.hpp"
#include "ImageProcessorModule.h"
#include"scc_motion.h"

class DetachDefectThreadSmartCroppingOfBags;

// 状态机
enum class RunningState
{
	Debug,
	Monitor,
	OpenRemoveFunc,
	Stop
};

class GlobalStructDataSmartCroppingOfBags
	:public QObject
{
	Q_OBJECT
public:
	std::unique_ptr<zwy::scc::Motion> motion;
public:
	std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > priorityQueue1;
	std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > priorityQueue2;
public:
	void build_PriorityQueue();
	void destroy_PriorityQueue();
public:
	DetachDefectThreadSmartCroppingOfBags *detachDefectThreadSmartCroppingOfBags;
public:
	void build_DetachDefectThreadSmartCroppingOfBags();
	void destroy_DetachDefectThreadSmartCroppingOfBags();
public slots:
	void onCameraReject(size_t index);

public:
	std::atomic<RunningState> runningState{ RunningState::Stop };

public:
	// 统计信息
	struct StatisticalInfo
	{
		std::atomic_uint64_t produceCount{ 0 };
		std::atomic_uint64_t wasteCount{ 0 };
		std::atomic<double> productionYield{ 0 };
		std::atomic<double> averageBagLength{ 0 };
		std::atomic_uint64_t produceCount1{ 0 };
		std::atomic_uint64_t produceCount2{ 0 };
	} statisticalInfo;

public:
	std::atomic_bool isTakePictures{ false };

public:
	static GlobalStructDataSmartCroppingOfBags& getInstance()
	{
		static GlobalStructDataSmartCroppingOfBags instance;
		return instance;
	}

	GlobalStructDataSmartCroppingOfBags(const GlobalStructDataSmartCroppingOfBags&) = delete;
	GlobalStructDataSmartCroppingOfBags& operator=(const GlobalStructDataSmartCroppingOfBags&) = delete;
private:
	GlobalStructDataSmartCroppingOfBags();
	~GlobalStructDataSmartCroppingOfBags();
public:
	void buildConfigManager(rw::oso::StorageType type);

	void buildImageProcessorModules(const QString& path);
	void destroyImageProcessingModule();

	std::unique_ptr<ImageProcessingModuleSmartCroppingOfBags> modelCamera1 = nullptr;
	std::unique_ptr<ImageProcessingModuleSmartCroppingOfBags> modelCamera2 = nullptr;


public:
	// 保存参数
	void buildImageSaveEngine();
	void destroyImageSaveEngine();
	std::unique_ptr<rw::rqw::ImageSaveEngine> imageSaveEngine{ nullptr };

	void saveGeneralConfig();
	void saveDlgProductSetConfig();
	void saveDlgProductScoreConfig();
	void saveDlgExposureTimeSetConfig();

public:
	// UI界面参数
	cdm::GeneralConfigSmartCroppingOfBags generalConfig;
	cdm::ScoreConfigSmartCroppingOfBags scoreConfig;
	cdm::SetConfigSmartCroppingOfBags setConfig;

public:
	void buildCamera();
	// 相机
	QString cameraIp1{ "11" };
	QString cameraIp2{ "12" };

	std::unique_ptr<rw::rqw::CameraPassiveThread> camera1{ nullptr };
	std::unique_ptr<rw::rqw::CameraPassiveThread> camera2{ nullptr };

	bool buildCamera1();
	bool buildCamera2();

	void destroyCamera();

	void destroyCamera1();
	void destroyCamera2();

	bool isTargetCamera(const QString& cameraIndex, const QString& targetName);
	rw::rqw::CameraMetaData cameraMetaDataCheck(const QString& cameraIndex, const QVector<rw::rqw::CameraMetaData>& cameraInfo);
	void setCameraExposureTime(int cameraIndex, size_t exposureTime);

	void setCameraDebugMod();
	void setCameraDefectMod();

public:
	std::unique_ptr<rw::oso::StorageContext> storeContext{ nullptr };
};
