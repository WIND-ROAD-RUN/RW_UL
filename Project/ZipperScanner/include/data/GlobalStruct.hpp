#pragma once

#include<memory>
#include<QString>
#include<QObject>

#include "GeneralConfig.hpp"
#include "ScoreConfig.hpp"
#include "SetConfig.hpp"
#include "oso_StorageContext.hpp"
#include "rqw_CameraObjectCore.hpp"
#include "rqw_CameraObjectThread.hpp"
#include "ZipperScannerDlgExposureTimeSet.hpp"
#include "ImageProcessorModule.h"
#include"dsl_PriorityQueue.hpp"
#include"Utilty.hpp"
#include<chrono>


class DetachDefectThreadZipper;

// 状态机
enum class RunningState
{
	Debug,
	Monitor,
	OpenRemoveFunc,
	Stop
};

enum class LightLevel {
	StrongLight, 
	MediumLight, 
	WeakLight
};

class GlobalStructDataZipper
	:public QObject
{
	Q_OBJECT
public:
	std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > priorityQueue1;
	std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > priorityQueue2;
public:
	void build_PriorityQueue();
	void destroy_PriorityQueue();
public:
	DetachDefectThreadZipper *detachDefectThreadZipper;
public:
	void build_DetachDefectThreadZipper();
	void destroy_DetachDefectThreadZipper();
public slots:
	void onCameraReject(size_t index);

public:
	std::atomic<RunningState> runningState{ RunningState::Stop };
	std::atomic<bool> debug_isDisplayRec{ true };
	std::atomic<bool> debug_isDisplayText{ true };

public:
	// 统计信息
	struct StatisticalInfo
	{
		std::atomic_uint64_t produceCount{ 0 };
		std::atomic_uint64_t wasteCount{ 0 };
		std::atomic<double> productionYield{ 0 };
		std::atomic<double> removeRate{ 0 };
		std::atomic_uint64_t produceCount1{ 0 };
		std::atomic_uint64_t produceCount2{ 0 };
	} statisticalInfo;

public:
	std::atomic_bool isTakePictures{ false };

public:
	static GlobalStructDataZipper& getInstance()
	{
		static GlobalStructDataZipper instance;
		return instance;
	}

	GlobalStructDataZipper(const GlobalStructDataZipper&) = delete;
	GlobalStructDataZipper& operator=(const GlobalStructDataZipper&) = delete;
private:
	GlobalStructDataZipper();
	~GlobalStructDataZipper() = default;
public:
	void setLightLevel(const LightLevel & level);
public:
	void buildConfigManager(rw::oso::StorageType type);

	void buildImageProcessorModules(const QString& path);
	void destroyImageProcessingModule();

	// 图像处理模块
	std::unique_ptr<ImageProcessingModuleZipper> modelCamera1 = nullptr;
	std::unique_ptr<ImageProcessingModuleZipper> modelCamera2 = nullptr;

	
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
	cdm::GeneralConfig generalConfig;
	cdm::ScoreConfig scoreConfig;
	cdm::SetConfig setConfig;

public:
	void buildCamera();
	// 相机
	QString cameraIp1{ "11" };
	QString cameraIp2{ "12" };

	std::unique_ptr<rw::rqw::CameraPassiveThread> camera1{ nullptr };
	std::unique_ptr<rw::rqw::CameraPassiveThread> camera2{ nullptr };

	cdm::ZipperScannerDlgExposureTimeSet dlgExposureTimeSetConfig{};

	bool buildCamera1();
	bool buildCamera2();

	void destroyCamera();

	void destroyCamera1();
	void destroyCamera2();

	bool isTargetCamera(const QString& cameraIndex, const QString& targetName);
	rw::rqw::CameraMetaData cameraMetaDataCheck(const QString& cameraIndex, const QVector<rw::rqw::CameraMetaData>& cameraInfo);
	void setCameraExposureTime(int cameraIndex, size_t exposureTime);

public:
	std::unique_ptr<rw::oso::StorageContext> storeContext{ nullptr };
};
