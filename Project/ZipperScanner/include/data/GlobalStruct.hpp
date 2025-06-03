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

class GlobalStructDataZipper
	:public QObject
{
	Q_OBJECT
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
	void buildConfigManager(rw::oso::StorageType type);
	


	// 保存参数
	void saveDlgProductSetConfig();
	void saveDlgProductScoreConfig();
	
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

	void destroyCamera1();
	void destroyCamera2();

	bool isTargetCamera(const QString& cameraIndex, const QString& targetName);
	rw::rqw::CameraMetaData cameraMetaDataCheck(const QString& cameraIndex, const QVector<rw::rqw::CameraMetaData>& cameraInfo);
	void setCameraExposureTime(int cameraIndex, size_t exposureTime);

public:
	std::unique_ptr<rw::oso::StorageContext> storeContext{ nullptr };


	
};
