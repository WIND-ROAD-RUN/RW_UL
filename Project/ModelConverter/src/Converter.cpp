#include"Converter.hpp"

Converter::Converter(QObject* parent) : QThread(parent)
{
	_processExportToEngine = new QProcess(this);
	connect(_processExportToEngine, &QProcess::readyReadStandardOutput,
		this, &Converter::handleOutput);
	connect(_processExportToEngine, &QProcess::readyReadStandardError,
		this, &Converter::handleError);
	connect(_processExportToEngine, &QProcess::finished,
		this, &Converter::handleFinished);
}

Converter::~Converter()
{
}

void Converter::run()
{
	std::string str = R"(.\trtexec.exe --onnx=)";
	str += inputFile.toStdString();
	str += R"( --saveEngine=)";
	str += outputFile.toStdString();
	_processExportToEngine->start("cmd.exe", { "/c",str.c_str() });

	exec();
}

void Converter::handleOutput()
{
	QByteArray output = _processExportToEngine->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void Converter::handleError()
{
	QByteArray output = _processExportToEngine->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void Converter::handleFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
	emit finish();
	quit();
}
