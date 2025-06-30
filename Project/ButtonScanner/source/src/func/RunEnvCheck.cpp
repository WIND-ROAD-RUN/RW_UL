#include"RunEnvCheck.hpp"

#include <QFile>
#include <QMessageBox>

#include "ButtonScannerDlgExposureTimeSet.hpp"
#include "ButtonScannerDlgHideScoreSet.hpp"
#include "ButtonScannerDlgProductSet.hpp"
#include "ButtonScannerMainWindow.hpp"
#include "ButtonScannerProduceLineSet.hpp"
#include"oso_StorageContext.hpp"
#include"ButtonUtilty.h"
#include "WarningIOSetConfig.hpp"
#include"cdm_ButtonScannerDlgWarningManager.h"

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
	try
	{
		if (QFile::exists(globalPath.exposureTimeSetConfigPath)) {
			cdm::ButtonScannerDlgExposureTimeSet config = *storageContext.load(globalPath.exposureTimeSetConfigPath.toStdString());
		}
	}
	catch (std::runtime_error & e)
	{
		QMessageBox::warning(nullptr, "提示", "ButtonScannerDlgExposureTimeSet配置已失效，请联系维护人员。");
		return false;

	}

	try
	{
		if (QFile::exists(globalPath.dlgHideScoreSetPath)) {
			cdm::DlgHideScoreSet config = *storageContext.load(globalPath.dlgHideScoreSetPath.toStdString());
		}
	}
	catch (std::runtime_error& e)
	{
		QMessageBox::warning(nullptr, "提示", "DlgHideScoreSet配置已失效，请联系维护人员。");
		return false;

	}

	try
	{
		if (QFile::exists(globalPath.dlgProduceLineSetConfigPath)) {
			cdm::ButtonScannerProduceLineSet config = *storageContext.load(globalPath.dlgProduceLineSetConfigPath.toStdString());
		}
	}
	catch (std::runtime_error& e)
	{
		QMessageBox::warning(nullptr, "提示", "ButtonScannerProduceLineSet配置已失效，请联系维护人员。");
		return false;

	}

	try
	{
		if (QFile::exists(globalPath.dlgProdutSetConfigPath)) {
			cdm::ButtonScannerDlgProductSet config = *storageContext.load(globalPath.dlgProdutSetConfigPath.toStdString());
		}
	}
	catch (std::runtime_error& e)
	{
		QMessageBox::warning(nullptr, "提示", "ButtonScannerDlgProductSet配置已失效，请联系维护人员。");
		return false;

	}

	try
	{
		if (QFile::exists(globalPath.mainWindowConfigPath)) {
			cdm::ButtonScannerMainWindow config = *storageContext.load(globalPath.mainWindowConfigPath.toStdString());
		}
	}
	catch (std::runtime_error& e)
	{
		QMessageBox::warning(nullptr, "提示", "ButtonScannerMainWindow配置已失效，请联系维护人员。");
		return false;

	}

	try
	{
		if (QFile::exists(globalPath.warningIOSetConfigPath)) {
			cdm::WarningIOSetConfig config = *storageContext.load(globalPath.warningIOSetConfigPath.toStdString());
		}
	}
	catch (std::runtime_error& e)
	{
		QMessageBox::warning(nullptr, "提示", "WarningIOSetConfig配置已失效，请联系维护人员。");
		return false;
	}

	try
	{
		if (QFile::exists(globalPath.warningManagerConfigPath)) {
			rw::cdm::ButtonScannerDlgWarningManager config = *storageContext.load(globalPath.warningManagerConfigPath.toStdString());
		}
	}
	catch (std::runtime_error& e)
	{
		QMessageBox::warning(nullptr, "提示", "ButtonScannerDlgWarningManager配置已失效，请联系维护人员。");
		return false;
	}



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

	if (!checkConfigIsOk())
	{
		return EnvCheckResult::EnvError; 
	}

	return EnvCheckResult::EnvOk;
}
