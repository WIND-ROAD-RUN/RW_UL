#include"stdafx.h"

#include"AiTrainModule.h"
#include"GlobalStruct.h"

AiTrainModule::AiTrainModule(QObject* parent)
	: QThread(parent) {
	auto enginePath = globalPath.modelRootPath + globalPath.engineFileName;
	auto namePath = globalPath.modelRootPath + globalPath.nameFileName;
	labelEngine = std::make_unique<rw::imeot::ModelEngineOT>(enginePath.toStdString(), namePath.toStdString());
	_processTrainModel = new QProcess();
	connect(_processTrainModel, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleTrainModelProcessOutput);
	connect(_processTrainModel, &QProcess::readyReadStandardError, this, &AiTrainModule::handleTrainModelProcessError);
	connect(_processTrainModel, &QProcess::finished, this, &AiTrainModule::handleTrainModelProcessFinished);

	_processExportModel = new QProcess();
	connect(_processExportModel, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleExportModelProcessOutput);
	connect(_processExportModel, &QProcess::readyReadStandardError, this, &AiTrainModule::handleExportModelProcessError);
	connect(_processExportModel, &QProcess::finished, this, &AiTrainModule::handleExportModelProcessFinished);
}

AiTrainModule::~AiTrainModule()
{
	wait();
	delete _processTrainModel;
	delete _processExportModel;
}

void AiTrainModule::startTrain()
{
	start();
}

rw::imeot::ProcessRectanglesResultOT AiTrainModule::getBody(
	std::vector<rw::imeot::ProcessRectanglesResultOT>& processRectanglesResult, bool& hasBody)
{
	hasBody = false;
	rw::imeot::ProcessRectanglesResultOT result;
	result.width = 0;
	result.height = 0;
	for (auto& i : processRectanglesResult)
	{
		if (i.classID == 0)
		{
			if ((i.width * i.height) > (result.width * result.height))
			{
				result = i;
				hasBody = true;
			}
		}
	}
	return result;
}

QVector<AiTrainModule::DataItem> AiTrainModule::getDataSet(const QVector<labelAndImg>& annotationDataSet, ModelType type, int classId)
{
	QVector<AiTrainModule::DataItem> result;
	switch (type)
	{
	case ModelType::Segment:
		result = getSegmentDataSet(annotationDataSet, classId);
		break;
	case ModelType::ObjectDetection:
		result = getObjectDetectionDataSet(annotationDataSet, classId);
		break;
	default:
		break;
	}
	return result;
}

QVector<AiTrainModule::DataItem> AiTrainModule::getSegmentDataSet(const QVector<labelAndImg>& annotationDataSet, int classId)
{
	QVector<AiTrainModule::DataItem> result;

	for (const auto& item : annotationDataSet)
	{
		std::string id = std::to_string(classId);

		// 归一化中心点和宽高
		double norCenterX = static_cast<double>(item.second.center_x) / static_cast<double>(_frameWidth);
		double norCenterY = static_cast<double>(item.second.center_y) / static_cast<double>(_frameHeight);
		double norWidth = static_cast<double>(item.second.width) / static_cast<double>(_frameWidth);
		double norHeight = static_cast<double>(item.second.height) / static_cast<double>(_frameHeight);

		// 计算椭圆上的 30 个点
		constexpr int numPoints = 30;
		std::string pointsStr;
		for (int i = 0; i < numPoints; ++i)
		{
			// 计算角度（均匀分布在 0 到 2π 之间）
			double angle = 2.0 * M_PI * i / numPoints;

			// 椭圆公式：x = centerX + a * cos(angle), y = centerY + b * sin(angle)
			double x = norCenterX + (norWidth / 2.0) * std::cos(angle);
			double y = norCenterY + (norHeight / 2.0) * std::sin(angle);

			// 将点添加到字符串中
			pointsStr += " " + std::to_string(x) + " " + std::to_string(y);
		}

		// 组合最终的标注字符串
		auto textStr = id + pointsStr;

		// 添加到结果集
		result.append({ item.first, QString::fromStdString(textStr) });
	}

	return result;
}

QVector<AiTrainModule::DataItem> AiTrainModule::getObjectDetectionDataSet(const QVector<labelAndImg>& annotationDataSet, int classId)
{
	QVector<AiTrainModule::DataItem> result;
	for (const auto& item : annotationDataSet)
	{
		std::string id = std::to_string(classId);
		//normalization归一化

		double norCenterX = static_cast<double>(item.second.center_x) / static_cast<double>(_frameWidth);
		double norCenterY = static_cast<double>(item.second.center_y) / static_cast<double>(_frameHeight);
		double norWidth = static_cast<double>(item.second.width) / static_cast<double>(_frameWidth);
		double norHeight = static_cast<double>(item.second.height) / static_cast<double>(_frameHeight);

		auto textStr = id + " " +
			std::to_string(norCenterX) + " " + std::to_string(norCenterY) + " "
			+ std::to_string(norWidth) + " " + std::to_string(norHeight);

		result.append({ item.first, QString::fromStdString(textStr) });
	}
	return result;
}

void AiTrainModule::clear_older_trainData()
{
	QString workPlace = globalPath.yoloV5RootPath;
	QDir dir(workPlace);
	QString absolutePath = dir.absolutePath();

	// 处理 Run 目录
	{
		QString runDir = workPlace + R"(runs\)";
		QString trainDir = runDir + R"(train\)";
		QString trainSeg = runDir + R"(train-seg\)";

		// 删除 trainDir 及其所有子文件和子文件夹
		QDir trainDirObj(trainDir);
		if (trainDirObj.exists()) {
			trainDirObj.removeRecursively();
		}

		// 删除 trainSeg 及其所有子文件和子文件夹
		QDir trainSegObj(trainSeg);
		if (trainSegObj.exists()) {
			trainSegObj.removeRecursively();
		}
	}

	// 处理 dataset 目录
	{
		QString dataSetDir = workPlace + R"(datasets\mydataset)";
		QString trainDir = dataSetDir + R"(\train\)";
		QString valDir = dataSetDir + R"(\val\)";
		QString testDir = dataSetDir + R"(\tes\)";

		// 删除 trainDir 及其所有子文件和子文件夹
		QDir trainDirObj(trainDir);
		if (trainDirObj.exists()) {
			trainDirObj.removeRecursively();
		}

		// 删除 valDir 及其所有子文件和子文件夹
		QDir valDirObj(valDir);
		if (valDirObj.exists()) {
			valDirObj.removeRecursively();
		}

		// 删除 testDir 及其所有子文件和子文件夹
		QDir testDirObj(testDir);
		if (testDirObj.exists()) {
			testDirObj.removeRecursively();
		}
	}

	//处理segDatasets
	{
		QString segDataSetDir = workPlace + R"(segDatasets\mydataset)";
		QString trainDir = segDataSetDir + R"(\train\)";
		QString valDir = segDataSetDir + R"(\val\)";
		QString testDir = segDataSetDir + R"(\tes\)";
		// 删除 trainDir 及其所有子文件和子文件夹
		QDir trainDirObj(trainDir);
		if (trainDirObj.exists()) {
			trainDirObj.removeRecursively();
		}
		// 删除 valDir 及其所有子文件和子文件夹
		QDir valDirObj(valDir);
		if (valDirObj.exists()) {
			valDirObj.removeRecursively();
		}
		// 删除 testDir 及其所有子文件和子文件夹
		QDir testDirObj(testDir);
		if (testDirObj.exists()) {
			testDirObj.removeRecursively();
		}
	}
	// 处理Temp下所有的onnx文件
	{
		QString tempDir = globalPath.modelStorageManagerTempPath;
		QDir tempDirObj(tempDir);
		if (!tempDirObj.exists()) {
			return;
		}

		// 获取所有 .onnx 文件
		QStringList filters;
		filters << "*.onnx";
		QStringList onnxFiles = tempDirObj.entryList(filters, QDir::Files);

		// 删除每个 .onnx 文件
		for (const QString& fileName : onnxFiles) {
			QString filePath = tempDirObj.filePath(fileName);
			if (!QFile::remove(filePath)) {
				qDebug() << "Failed to remove file:" << filePath;
			}
			else {
				qDebug() << "Removed file:" << filePath;
			}
		}
	}
}

void AiTrainModule::copyTrainData(const QVector<AiTrainModule::DataItem>& dataSet)
{
	if (_modelType == ModelType::ObjectDetection)
	{
		copyTrainImgData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\tes\)"));
		copyTrainImgData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\train\images\)"));
		copyTrainImgData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\val\images\)"));

		copyTrainLabelData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\train\labels)"));
		copyTrainLabelData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\val\labels)"));
	}
	else if (_modelType == ModelType::Segment)
	{
		copyTrainImgData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\tes\)"));
		copyTrainImgData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\train\images\)"));
		copyTrainImgData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\val\images\)"));

		copyTrainLabelData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\train\labels)"));
		copyTrainLabelData(dataSet, QString(globalPath.yoloV5RootPath + R"(\datasets\mydataset\val\labels)"));
	}
}

void AiTrainModule::copyTrainImgData(const QVector<AiTrainModule::DataItem>& dataSet, const QString& path)
{
	// 检查目标路径是否存在，如果不存在则创建
	QDir dir(path);
	if (!dir.exists()) {
		if (!dir.mkpath(path)) {
			emit appRunLog("Failed to create directory: " + path);
			return;
		}
	}

	// 遍历 dataSet 并拷贝图片
	for (const auto& item : dataSet) {
		QString sourcePath = item.first; // 图片的源路径
		QString fileName = QFileInfo(sourcePath).fileName(); // 获取文件名
		QString destinationPath = path + QDir::separator() + fileName; // 目标路径

		// 拷贝文件
		if (!QFile::copy(sourcePath, destinationPath)) {
			emit appRunLog("Failed to copy file: " + sourcePath + " to " + destinationPath);
		}
		else {
			emit appRunLog("Copied file: " + sourcePath + " to " + destinationPath);
		}
	}
}

void AiTrainModule::copyTrainLabelData(const QVector<AiTrainModule::DataItem>& dataSet, const QString& path)
{
	// 检查目标路径是否存在，如果不存在则创建
	QDir dir(path);
	if (!dir.exists()) {
		if (!dir.mkpath(path)) {
			emit appRunLog("Failed to create directory: " + path);
			return;
		}
	}

	// 遍历 dataSet 并保存 label 数据
	for (const auto& item : dataSet) {
		QString fileName = QFileInfo(item.first).baseName() + ".txt"; // 获取文件名并添加 .txt 后缀
		QString filePath = path + QDir::separator() + fileName; // 目标文件路径

		QFile file(filePath);
		if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
			QTextStream out(&file);
			out << item.second; // 写入 label 数据
			file.close();
			emit appRunLog("Saved label file: " + filePath);
		}
		else {
			emit appRunLog("Failed to save label file: " + filePath);
		}
	}
}

void AiTrainModule::trainSegmentModel()
{
	auto str = "activate yolov5 && cd /d " + globalPath.yoloV5RootPath.toStdString() + R"(segment\)" + " && python train.py";
	_processTrainModel->start("cmd.exe", { "/c",str.c_str() });
}

void AiTrainModule::trainObbModel()
{
	auto str = "activate yolov5 && cd /d " + globalPath.yoloV5RootPath.toStdString() + " && python train.py";
	_processTrainModel->start("cmd.exe", { "/c",str.c_str() });
}

void AiTrainModule::exportOnnxModel()
{
	if (_modelType == ModelType::Segment)
	{
		auto str = "activate yolov5 && cd /d " + globalPath.yoloV5RootPath.toStdString() + " && python export_seg.py";
		_processExportModel->start("cmd.exe", { "/c",str.c_str() });
	}
	else if (_modelType == ModelType::ObjectDetection)
	{
		auto str = "activate yolov5 && cd /d " + globalPath.yoloV5RootPath.toStdString() + " && python export.py";
		_processExportModel->start("cmd.exe", { "/c",str.c_str() });
	}
}

void AiTrainModule::copyModelToTemp()
{
	// 源文件路径
	QString sourceFilePath = globalPath.yoloV5RootPath + R"(runs\train\exp\weights\best.onnx)";

	if (_modelType == ModelType::Segment)
	{
		sourceFilePath = globalPath.yoloV5RootPath + R"(runs\train-seg\exp\weights\best.onnx)";
	}
	else if (_modelType == ModelType::ObjectDetection)
	{
		sourceFilePath = globalPath.yoloV5RootPath + R"(runs\train\exp\weights\best.onnx)";
	}

	// 目标目录路径
	QString targetDirectory = globalPath.modelStorageManagerTempPath;

	// 确保目标目录存在，如果不存在则创建
	QDir dir(targetDirectory);
	if (!dir.exists()) {
		if (!dir.mkpath(targetDirectory)) {
			emit appRunLog("Failed to create target directory: " + targetDirectory);
			return;
		}
	}

	QString targetFilePath = targetDirectory + "modelOnnx.onnx";

	if (_modelType == ModelType::Segment)
	{
		targetFilePath = targetDirectory + "customSO.onnx";
	}
	else if (_modelType == ModelType::ObjectDetection)
	{
		targetFilePath = targetDirectory + "customOO.onnx";
	}

	if (QFile::exists(targetFilePath)) {
		if (!QFile::remove(targetFilePath)) {
			emit appRunLog("Failed to remove existing file:" + targetFilePath);
		}
	}

	// 拷贝文件并重命名
	if (QFile::copy(sourceFilePath, targetFilePath)) {
		emit appRunLog("File copied and renamed successfully: " + targetFilePath);
	}
	else {
		emit appRunLog("Failed to copy and rename file: " + sourceFilePath);
	}
}

void AiTrainModule::packageModelToStorage()
{
	// 获取当前日期和时间
	QDateTime currentDateTime = QDateTime::currentDateTime();
	// 格式化为 "yyyyMMddHHmmss" 格式
	QString formattedDateTime = currentDateTime.toString("yyyyMMddHHmmss");

	auto storage = globalPath.modelStorageManagerRootPath + formattedDateTime + R"(\)";
	QString sourceFilePath = globalPath.modelStorageManagerTempPath;

	copy_all_files_to_storage(sourceFilePath, storage);

	auto& global = GlobalStructData::getInstance();

	rw::cdm::AiModelConfig config;
	std::hash<std::string> hasher;
	config.id = static_cast<long>(hasher(formattedDateTime.toStdString()));
	config.name = formattedDateTime.toStdString();
	if (_modelType == ModelType::ObjectDetection)
	{
		config.modelType = rw::cdm::ModelType::BladeShape;
	}
	else if (_modelType == ModelType::Segment)
	{
		config.modelType = rw::cdm::ModelType::Color;
	}
	else
	{
		config.modelType = rw::cdm::ModelType::Undefined;
	}
	config.sideLight = global.mainWindowConfig.sideLight;
	config.upLight = global.mainWindowConfig.upLight;
	config.downLight = global.mainWindowConfig.downLight;
	config.exposureTime = global.dlgExposureTimeSetConfig.expousureTime;
	if (config.exposureTime <= 200)
	{
		config.gain = 0;
	}
	else
	{
		config.gain = 5;
	}
	config.date = formattedDateTime.toStdString();
	config.rootPath = storage.toStdString();

	std::string savePath = storage.toStdString() + formattedDateTime.toStdString() + ".xml";

	global.storeContext->save(config, savePath);

	global.modelStorageManager->addModelConfig(config);

	global.modelStorageManager->saveIndexConfig();

	emit appRunLog("模型打包完成");
	emit updateTrainTitle("模型打包完成");
}

void AiTrainModule::copy_all_files_to_storage(const QString& sourceFilePath, const QString& storage)
{
	// 确保目标路径存在，如果不存在则创建
	QDir storageDir(storage);
	if (!storageDir.exists()) {
		if (!storageDir.mkpath(storage)) {
			emit appRunLog("Failed to create storage directory: " + storage);
			return;
		}
	}

	// 获取源目录
	QDir sourceDir(sourceFilePath);
	if (!sourceDir.exists()) {
		emit appRunLog("Source directory does not exist: " + sourceFilePath);
		return;
	}

	// 获取源目录中的所有文件
	QStringList files = sourceDir.entryList(QDir::Files);
	for (const QString& fileName : files) {
		QString sourceFile = sourceDir.filePath(fileName); // 源文件完整路径
		QString targetFile = storageDir.filePath(fileName); // 目标文件完整路径

		// 拷贝文件
		if (QFile::copy(sourceFile, targetFile)) {
			emit appRunLog("Copied file: " + sourceFile + " to " + targetFile);
		}
		else {
			emit appRunLog("Failed to copy file: " + sourceFile + " to " + targetFile);
		}
	}

	// 获取源目录中的所有子目录
	QStringList directories = sourceDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& dirName : directories) {
		QString sourceSubDir = sourceDir.filePath(dirName); // 源子目录完整路径
		QString targetSubDir = storageDir.filePath(dirName); // 目标子目录完整路径

		// 递归拷贝子目录
		copy_all_files_to_storage(sourceSubDir, targetSubDir);
	}
}

cv::Mat AiTrainModule::getMatFromPath(const QString& path)
{
	cv::Mat image = cv::imread(path.toStdString());
	if (image.empty()) {
		qDebug() << "Failed to load image:" << path;
	}
	return image;
}

void AiTrainModule::run()
{
	auto& global = GlobalStructData::getInstance();
	global.isTrainModel = true;
	emit updateTrainState(true);
	emit updateTrainTitle("正在训练");
	emit appRunLog("训练启动....");

	emit appRunLog("清理旧的训练数据....");
	clear_older_trainData();

	//获取图片的label
	auto annotationGoodDataSet = annotation_data_set(false);
	auto annotationBadDataSet = annotation_data_set(true);
	auto dataSet = getDataSet(annotationGoodDataSet, _modelType, 1);
	auto dataSetBad = getDataSet(annotationBadDataSet, _modelType, 0);
	QString GoodSetLog = "其中正确的纽扣数据集有" + QString::number(dataSet.size()) + "条数据";
	QString BadSetLog = "其中错误的纽扣数据集有" + QString::number(dataSetBad.size()) + "条数据";
	emit appRunLog(GoodSetLog);
	emit appRunLog(BadSetLog);

	//拷贝训练数据
	emit appRunLog("拷贝训练文件");
	copyTrainData(dataSet);
	copyTrainData(dataSetBad);

	if (_modelType == ModelType::Segment)
	{
		emit appRunLog("开始训练分割模型");
		trainSegmentModel();
	}
	else if (_modelType == ModelType::ObjectDetection)
	{
		emit appRunLog("开始训练检测模型");
		trainObbModel();
	}

	exec();
}

QVector<AiTrainModule::labelAndImg> AiTrainModule::annotation_data_set(bool isBad)
{
	QVector<QString> imageList;
	if (isBad)
	{
		emit appRunLog("正在标注要筛选的纽扣数据集");
		imageList = GlobalStructData::getInstance().modelStorageManager->getBadImagePathList();
	}
	else
	{
		emit appRunLog("正在标注正确的纽扣的数据集");
		imageList = GlobalStructData::getInstance().modelStorageManager->getGoodImagePathList();
	}

	int i = 0;

	QVector<labelAndImg> dataSet;
	dataSet.reserve(100);

	//获取图片的label
	for (const auto& imagePath : imageList) {
		auto image = getMatFromPath(imagePath);
		if (image.empty()) {
			continue;
		}
		_frameWidth = image.cols;
		_frameHeight = image.rows;
		cv::Mat resultMat;
		std::vector<rw::imeot::ProcessRectanglesResultOT> result;
		labelEngine->ProcessMask(image, resultMat, result);
		QString log = QString::number(i) + " ";

		bool hasBody;
		auto body = getBody(result, hasBody);
		if (!hasBody)
		{
			continue;
		}

		dataSet.emplaceBack(imagePath, body);
		log += "ClassId: " + QString::number(body.classID) + " center_x" + QString::number(body.center_x) + " center_y" + QString::number(body.center_y);
		emit appRunLog(log);
		i++;
	}
	emit appRunLog("标注完" + QString::number(dataSet.size()) + "条数据");

	return dataSet;
}

void AiTrainModule::handleTrainModelProcessOutput()
{
	QByteArray output = _processTrainModel->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleTrainModelProcessError()
{
	QByteArray errorOutput = _processTrainModel->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
	int total = 100;
	int complete = -1;
	if (_modelType == ModelType::ObjectDetection) {
		complete = parseProgressOO(errorStr, total);
	}
	else if (_modelType == ModelType::Segment)
	{
		complete = parseProgressSO(errorStr, total);
	}

	if (complete == -1)
	{
		return;
	}
	emit updateProgress(complete, 100);
}

void AiTrainModule::handleTrainModelProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		exportOnnxModel();
	}
	else
	{
		emit updateTrainTitle("训练失败");
		emit updateTrainState(false);
		auto& global = GlobalStructData::getInstance();
		global.isTrainModel = false;
		quit();
	}
}

void AiTrainModule::handleExportModelProcessOutput()
{
	QByteArray output = _processExportModel->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleExportModelProcessError()
{
	QByteArray errorOutput = _processExportModel->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
}

void AiTrainModule::handleExportModelProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("导出完成");
		copyModelToTemp();
		packageModelToStorage();
		emit updateProgress(100, 100);
		emit updateTrainState(false);
	}
	else
	{
		emit updateTrainTitle("导出失败");
		emit updateTrainState(false);
	}
	auto& global = GlobalStructData::getInstance();
	global.isTrainModel = false;
	quit();
}

void AiTrainModule::cancelTrain()
{
	if (_processTrainModel->state() == QProcess::Running) {
		_processTrainModel->kill();
		_processTrainModel->waitForFinished();
	}
	if (_processExportModel->state() == QProcess::Running) {
		_processExportModel->kill();
		_processExportModel->waitForFinished();
	}
	emit updateTrainTitle("训练已取消");
	emit updateTrainState(false);
	quit();
}

int AiTrainModule::parseProgressOO(const QString& logText, int& totalTasks) {
	// 匹配整行日志的正则表达式
	QRegularExpression lineRegex(R"((\d+/\d+)\s+\d+\.\d+G\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+:)");
	QRegularExpressionMatch lineMatch = lineRegex.match(logText);

	if (lineMatch.hasMatch()) {
		// 提取类似 "8/99" 的部分
		QString progress = lineMatch.captured(1);

		// 再次用正则表达式解析 "8/99"
		QRegularExpression progressRegex(R"((\d+)/(\d+))");
		QRegularExpressionMatch progressMatch = progressRegex.match(progress);

		if (progressMatch.hasMatch()) {
			int completedTasks = progressMatch.captured(1).toInt();
			totalTasks = progressMatch.captured(2).toInt();
			return completedTasks;
		}
	}

	// 如果未匹配到，返回 -1 表示解析失败
	totalTasks = -1;
	return -1;
}

int AiTrainModule::parseProgressSO(const QString& logText, int& totalTasks) {
	// 匹配整行日志的正则表达式
	QRegularExpression lineRegex(R"((\d+/\d+)\s+\d+\.\d+G\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+\d+\s+\d+:)");
	QRegularExpressionMatch lineMatch = lineRegex.match(logText);

	if (lineMatch.hasMatch()) {
		// 提取类似 "3/99" 的部分
		QString progress = lineMatch.captured(1);

		// 再次用正则表达式解析 "3/99"
		QRegularExpression progressRegex(R"((\d+)/(\d+))");
		QRegularExpressionMatch progressMatch = progressRegex.match(progress);

		if (progressMatch.hasMatch()) {
			int completedTasks = progressMatch.captured(1).toInt();
			totalTasks = progressMatch.captured(2).toInt();
			return completedTasks;
		}
	}

	// 如果未匹配到，返回 -1 表示解析失败
	totalTasks = -1;
	return -1;
}