#pragma once
#include <QProcess>
#include <qsharedmemory.h>
#include <QString>

enum class EnvCheckResult
{
	EnvOk,
	EnvError
};

class RunEnvCheck
{
private:
	static bool isSingleInstance(const QString& instanceName);

	static bool isProcessRunning(const QString& processName);

	static bool checkConfigIsOk();
	
public:
	static EnvCheckResult envCheck();

};
