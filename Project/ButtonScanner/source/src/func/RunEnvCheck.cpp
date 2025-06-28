#include"RunEnvCheck.hpp"

#include <QMessageBox>

#include "ButtonScannerDlgExposureTimeSet.hpp"
#include"oso_StorageContext.hpp"
#include"ButtonUtilty.h"

bool RunEnvCheck::isSingleInstance(const QString& instanceName)
{
	static QSharedMemory sharedMemory(instanceName);
	if (!sharedMemory.create(1)) {
		return false; // 已有实例运行
	}
	return true; // 当前实例是唯一的
}

bool RunEnvCheck::isProcessRunning(const QString& processName)
{
	QProcess process;
	process.start("tasklist", QStringList() << "/FI" << QString("IMAGENAME eq %1").arg(processName));
	process.waitForFinished();
	QString output = process.readAllStandardOutput();
	return output.contains(processName, Qt::CaseInsensitive);
}

bool RunEnvCheck::checkConfigIsOk()
{
	rw::oso::StorageContext storageContext(rw::oso::StorageType::Xml);
	cdm::ButtonScannerDlgExposureTimeSet config = *storageContext.load(globalPath.configRootPath.toStdString());

	return true;
}

EnvCheckResult RunEnvCheck::envCheck()
{
	// 检查海康威视软件是否运行
	if (isProcessRunning("MVS.exe")) {
		QMessageBox::warning(nullptr, "提示", "检测到海康威视软件正在运行，请先关闭后再启动本程序。");
		return EnvCheckResult::EnvError;
	}

	// 检查度申软件是否运行
	if (isProcessRunning("BasedCam3.exe")) {
		QMessageBox::warning(nullptr, "提示", "检测到度申相机平台软件正在运行，请先关闭后再启动本程序。");
		return EnvCheckResult::EnvError;
	}

	// 检查单实例
	const QString instanceName = "ButtonScannerInstance";
	if (!isSingleInstance(instanceName)) {
		QMessageBox::critical(nullptr, "错误", "程序已在运行中，无法启动多个实例。");
		return EnvCheckResult::EnvError; // 退出程序
	}

	return EnvCheckResult::EnvOk;
}
