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
#include"DetachDefectThread.h"
#include "DetachUtiltyThread.h"
#include"MonitorIO.h"
#include"rqw_ZMotion.hpp"
#include"rqw_MonitorMotionIO.hpp"

#include"scc_motion.h"


enum class RunningState
{
	Debug,
	openRemove,
	Stop
};

enum class RemoveState
{
	SmartCrop,
	PrintingInspection
};

class GlobalStructThreadSmartCroppingOfBags
	:public QObject
{
	Q_OBJECT
public:
	static GlobalStructThreadSmartCroppingOfBags& getInstance()
	{
		static GlobalStructThreadSmartCroppingOfBags instance;
		return instance;
	}

	GlobalStructThreadSmartCroppingOfBags(const GlobalStructThreadSmartCroppingOfBags&) = delete;
	GlobalStructThreadSmartCroppingOfBags& operator=(const GlobalStructThreadSmartCroppingOfBags&) = delete;

private:
	GlobalStructThreadSmartCroppingOfBags()=default;
	~GlobalStructThreadSmartCroppingOfBags() = default;
public:
	std::unique_ptr<DetachUtiltyThreadSmartCroppingOfBags> _detachUtiltyThreadSmartCroppingOfBags{ nullptr };
	std::atomic_bool _isUpdateMonitoyInfo{false};
public:
	std::unique_ptr<MonitorIOSmartCroppingOfBags> monitorIOSmartCroppingOfBags{ nullptr };
	std::unique_ptr<DetachDefectThreadSmartCroppingOfBags> detachDefectThreadSmartCroppingOfBags{ nullptr };
	std::unique_ptr<rw::rqw::MonitorZMotionIOStateThread> monitorZMotionIOStateThread{ nullptr };
public:
	void build_detachThread();
	void destroy_detachThread();
	void start_detachThread();
public slots:
	void getQieDaoDI(size_t index, bool state);
signals:
	void appendPulse(double currentPulse);
private:
	bool _qiedaoPre{false};
	bool _qieDaoLast{ false };
public:
	std::atomic_bool isQieDao{false};
	std::atomic<Time> currentQieDaoTime;
};

class GlobalStructDataSmartCroppingOfBags
	:public QObject
{
	Q_OBJECT
public:
	rw::rqw::ZMotion zMotion;
public:
	bool build_motion();
	void destroy_motion();
public:
	std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > priorityQueue1;
	std::unique_ptr<rw::dsl::ThreadSafeDHeap<Time, Time> > priorityQueue2;
public:
	void build_PriorityQueue();
	void destroy_PriorityQueue();

public:
	std::atomic<RunningState> runningState{ RunningState::Stop };
	std::atomic<RemoveState> removeState{ RemoveState::SmartCrop };
public:
	// 统计信息
	struct StatisticalInfo
	{
		std::atomic_uint64_t produceCount{ 0 };
		std::atomic_uint64_t wasteCount{ 0 };
		std::atomic<double> productionYield{ 0 };
		std::atomic<double> averageBagLength{ 0 };
	} statisticalInfo;

public:
	std::atomic_bool isTakePictures{ false };
	std::atomic_bool isViewIO{ false };

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
	void buildImageSaveEngine();
	void destroyImageSaveEngine();
	std::unique_ptr<rw::rqw::ImageSaveEngine> imageSaveEngine{ nullptr };

	void saveGeneralConfig();
	void saveDlgProductSetConfig();
	void saveDlgProductScoreConfig();

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

	double pulse{ 0 };

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
