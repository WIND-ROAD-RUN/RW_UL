#include"stdafx.h"

#include"AiTrainModule.h"
#include"GlobalStruct.h"

AiTrainModule::AiTrainModule(QObject* parent)
	: QThread(parent) {
	auto enginePath = globalPath.modelRootPath + globalPath.engineSeg;
	rw::ModelEngineConfig config;
	config.modelPath = enginePath.toStdString();
	labelEngine = rw::ModelEngineFactory::createModelEngine(config, rw::ModelType::Yolov11_Seg, rw::ModelEngineDeployType::TensorRT);
	_processTrainModelBladeShape = new QProcess();
	_processExportToEngine = new QProcess();
	connect(_processTrainModelBladeShape, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleProcessTrainModelBladeShapeOutput);
	connect(_processTrainModelBladeShape, &QProcess::readyReadStandardError, this, &AiTrainModule::handleProcessTrainModelBladeShapeError);
	connect(_processTrainModelBladeShape, &QProcess::finished, this, &AiTrainModule::handleProcessTrainModelBladeShapeFinished);

	connect(_processExportToEngine, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleProcessExportEngineBladeShapeOutput);
	connect(_processExportToEngine, &QProcess::readyReadStandardError, this, &AiTrainModule::handleProcessExportEngineBladeShapeError);
	connect(_processExportToEngine, &QProcess::finished, this, &AiTrainModule::handleProcessExportEngineBladeShapeFinished);

	_processTrainModelColor1 = new QProcess();
	connect(_processTrainModelColor1, &QProcess::readyReadStandardOutput,
		this, &AiTrainModule::handleProcessTrainModelColor1Output);
	connect(_processTrainModelColor1, &QProcess::readyReadStandardError,
		this, &AiTrainModule::handleProcessTrainModelColor1Error);
	connect(_processTrainModelColor1, &QProcess::finished,
		this, &AiTrainModule::handleProcessTrainModelColor1Finished);

	_processTrainModelColor2 = new QProcess();
	connect(_processTrainModelColor2, &QProcess::readyReadStandardOutput,
		this, &AiTrainModule::handleProcessTrainModelColor2Output);
	connect(_processTrainModelColor2, &QProcess::readyReadStandardError,
		this, &AiTrainModule::handleProcessTrainModelColor2Error);
	connect(_processTrainModelColor2, &QProcess::finished,
		this, &AiTrainModule::handleProcessTrainModelColor2Finished);

	_processTrainModelColor3 = new QProcess();
	connect(_processTrainModelColor3, &QProcess::readyReadStandardOutput,
		this, &AiTrainModule::handleProcessTrainModelColor3Output);
	connect(_processTrainModelColor3, &QProcess::readyReadStandardError,
		this, &AiTrainModule::handleProcessTrainModelColor3Error);
	connect(_processTrainModelColor3, &QProcess::finished,
		this, &AiTrainModule::handleProcessTrainModelColor3Finished);

	_processTrainModelColor4 = new QProcess();
	connect(_processTrainModelColor4, &QProcess::readyReadStandardOutput,
		this, &AiTrainModule::handleProcessTrainModelColor4Output);
	connect(_processTrainModelColor4, &QProcess::readyReadStandardError,
		this, &AiTrainModule::handleProcessTrainModelColor4Error);
	connect(_processTrainModelColor4, &QProcess::finished,
		this, &AiTrainModule::handleProcessTrainModelColor4Finished);

	_processExportToEngineColor1 = new QProcess();
	connect(_processExportToEngineColor1, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleProcessExportEngineColor1Output);
	connect(_processExportToEngineColor1, &QProcess::readyReadStandardError, this, &AiTrainModule::handleProcessExportEngineColor1Error);
	connect(_processExportToEngineColor1, &QProcess::finished, this, &AiTrainModule::handleProcessExportEngineColor1Finished);

	_processExportToEngineColor2 = new QProcess();
	connect(_processExportToEngineColor2, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleProcessExportEngineColor2Output);
	connect(_processExportToEngineColor2, &QProcess::readyReadStandardError, this, &AiTrainModule::handleProcessExportEngineColor2Error);
	connect(_processExportToEngineColor2, &QProcess::finished, this, &AiTrainModule::handleProcessExportEngineColor2Finished);

	_processExportToEngineColor3 = new QProcess();
	connect(_processExportToEngineColor3, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleProcessExportEngineColor3Output);
	connect(_processExportToEngineColor3, &QProcess::readyReadStandardError, this, &AiTrainModule::handleProcessExportEngineColor3Error);
	connect(_processExportToEngineColor3, &QProcess::finished, this, &AiTrainModule::handleProcessExportEngineColor3Finished);

	_processExportToEngineColor4 = new QProcess();
	connect(_processExportToEngineColor4, &QProcess::readyReadStandardOutput, this, &AiTrainModule::handleProcessExportEngineColor4Output);
	connect(_processExportToEngineColor4, &QProcess::readyReadStandardError, this, &AiTrainModule::handleProcessExportEngineColor4Error);
	connect(_processExportToEngineColor4, &QProcess::finished, this, &AiTrainModule::handleProcessExportEngineColor4Finished);
}

AiTrainModule::~AiTrainModule()
{
	cancelTrain();
	wait();
	delete _processTrainModelBladeShape;
}

void AiTrainModule::startTrain()
{
	start();
}

QVector<AiTrainModule::DataItem> AiTrainModule::getDataSet(const QVector<labelAndImg>& annotationDataSet, int classId)
{
	QVector<AiTrainModule::DataItem> result;
	result = getObjectDetectionDataSet(annotationDataSet, classId);
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
	QString workPlace = globalPath.trainAIRootPath;

	QString runsPath = R"(./runs)";
	// 删除 runs 目录及其所有子文件和子文件夹
	QDir dir(runsPath);
	QString absolutePath = dir.absolutePath();
	if (dir.exists()) {
		dir.removeRecursively();
	}
	// 处理 obb 目录
	{
		QString obbDir = workPlace + R"(\Obb\)";
		// 删除 obb 及其所有子文件和子文件夹
		QDir obbDirObj(obbDir);
		if (obbDirObj.exists()) {
			obbDirObj.removeRecursively();
		}
	}

	//处理seg
	{
		QString trainDir = workPlace + R"(\Seg\images\)";
		QString LabelDir = workPlace + R"(\Seg\labels\)";

		// 删除 trainDir 及其所有子文件和子文件夹
		QDir trainDirObj(trainDir);
		if (trainDirObj.exists()) {
			trainDirObj.removeRecursively();
		}

		// 删除 LabelDir 及其所有子文件和子文件夹
		QDir testDirObj(LabelDir);
		if (testDirObj.exists()) {
			testDirObj.removeRecursively();
		}
	}
	// 处理Temp下所有的onnx和engine文件
	{
		QString tempDir = globalPath.modelStorageManagerTempPath;
		QDir tempDirObj(tempDir);
		if (!tempDirObj.exists()) {
			return;
		}

		// 获取所有 .onnx 和 .engine 文件
		QStringList filters;
		filters << "*.onnx" << "*.engine";
		QStringList files = tempDirObj.entryList(filters, QDir::Files);

		// 删除每个文件
		for (const QString& fileName : files) {
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

void AiTrainModule::clear_older_trainData_color()
{
	QString workPlace = globalPath.trainAIRootPath;

	QString runsPath = R"(./runs)";
	// 删除 runs 目录及其所有子文件和子文件夹
	QDir dir(runsPath);
	QString absolutePath = dir.absolutePath();
	if (dir.exists()) {
		dir.removeRecursively();
	}
	// 处理 obb 目录
	{
		QString obbDir = workPlace + R"(\Obb\)";
		// 删除 obb 及其所有子文件和子文件夹
		QDir obbDirObj(obbDir);
		if (obbDirObj.exists()) {
			obbDirObj.removeRecursively();
		}
	}

	//处理seg
	{
		QString trainDir = workPlace + R"(\Seg\images\)";
		QString LabelDir = workPlace + R"(\Seg\labels\)";

		// 删除 trainDir 及其所有子文件和子文件夹
		QDir trainDirObj(trainDir);
		if (trainDirObj.exists()) {
			trainDirObj.removeRecursively();
		}

		// 删除 LabelDir 及其所有子文件和子文件夹
		QDir testDirObj(LabelDir);
		if (testDirObj.exists()) {
			testDirObj.removeRecursively();
		}
	}
}

void AiTrainModule::copyTrainData(const QVector<AiTrainModule::DataItem>& dataSet)
{
	copyTrainImgData(dataSet, QString(globalPath.trainAIRootPath + R"(\Obb\images\)"));

	copyTrainLabelData(dataSet, QString(globalPath.trainAIRootPath + R"(\Obb\labels\)"));
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

void AiTrainModule::trainColorModel(int index)
{
	if (index == 1)
	{
		std::string str = "activate yolov11 && python ./train_yolov11_obb_Color.py";
		_processTrainModelColor1->start("cmd.exe", { "/c",str.c_str() });
	}
	if (index == 2)
	{
		std::string str = "activate yolov11 && python ./train_yolov11_obb_Color.py";
		_processTrainModelColor2->start("cmd.exe", { "/c",str.c_str() });
	}
	if (index == 3)
	{
		std::string str = "activate yolov11 && python ./train_yolov11_obb_Color.py";
		_processTrainModelColor3->start("cmd.exe", { "/c",str.c_str() });
	}
	if (index == 4)
	{
		std::string str = "activate yolov11 && python ./train_yolov11_obb_Color.py";
		_processTrainModelColor4->start("cmd.exe", { "/c",str.c_str() });
	}
}

void AiTrainModule::trainShapeModel()
{
	//conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda ultralytics
	std::string str = "activate yolov11 && python ./train_yolov11_obb_shape.py";
	_processTrainModelBladeShape->start("cmd.exe", { "/c",str.c_str() });
}

void AiTrainModule::copyModelToTemp()
{
	QString exePath = QCoreApplication::applicationDirPath();
	QString sourceFilePath = exePath + R"(/runs)";

	sourceFilePath = sourceFilePath + R"(/detect/train/weights/best.onnx)";

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

	QString targetFilePath1 = targetDirectory + "customOO1.onnx";
	QString targetFilePath2 = targetDirectory + "customOO2.onnx";
	QString targetFilePath3 = targetDirectory + "customOO3.onnx";
	QString targetFilePath4 = targetDirectory + "customOO4.onnx";

	// 拷贝文件并重命名
	if (QFile::exists(targetFilePath1)) {
		if (!QFile::remove(targetFilePath1)) {
			emit appRunLog("Failed to remove existing file:" + targetFilePath1);
		}
	}

	if (QFile::copy(sourceFilePath, targetFilePath1)) {
		emit appRunLog("File copied and renamed successfully: " + targetFilePath1);
	}
	else {
		emit appRunLog("Failed to copy and rename file: " + sourceFilePath);
	}

	// 拷贝文件并重命名
	if (QFile::exists(targetFilePath2)) {
		if (!QFile::remove(targetFilePath2)) {
			emit appRunLog("Failed to remove existing file:" + targetFilePath2);
		}
	}

	if (QFile::copy(sourceFilePath, targetFilePath2)) {
		emit appRunLog("File copied and renamed successfully: " + targetFilePath2);
	}
	else {
		emit appRunLog("Failed to copy and rename file: " + sourceFilePath);
	}

	// 拷贝文件并重命名
	if (QFile::exists(targetFilePath3)) {
		if (!QFile::remove(targetFilePath3)) {
			emit appRunLog("Failed to remove existing file:" + targetFilePath3);
		}
	}

	if (QFile::copy(sourceFilePath, targetFilePath3)) {
		emit appRunLog("File copied and renamed successfully: " + targetFilePath3);
	}
	else {
		emit appRunLog("Failed to copy and rename file: " + sourceFilePath);
	}

	// 拷贝文件并重命名
	if (QFile::exists(targetFilePath4)) {
		if (!QFile::remove(targetFilePath4)) {
			emit appRunLog("Failed to remove existing file:" + targetFilePath4);
		}
	}

	if (QFile::copy(sourceFilePath, targetFilePath4)) {
		emit appRunLog("File copied and renamed successfully: " + targetFilePath4);
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
	if (_modelType == ModelType::BladeShape)
	{
		config.modelType = rw::cdm::ModelType::BladeShape;
	}
	else if (_modelType == ModelType::Color)
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

void AiTrainModule::copyModelToTempColor(int workIndex)
{
	QString exePath = QCoreApplication::applicationDirPath();
	QString sourceFilePath = exePath + R"(/runs)";

	sourceFilePath = sourceFilePath + R"(/detect/train/weights/best.onnx)";

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

	QString targetFilePath;

	if (workIndex == 1)
	{
		targetFilePath = targetDirectory + "customOO1.onnx";
	}
	else if (workIndex == 2)
	{
		targetFilePath = targetDirectory + "customOO2.onnx";
	}
	else if (workIndex == 3)
	{
		targetFilePath = targetDirectory + "customOO3.onnx";
	}
	else if (workIndex == 4)
	{
		targetFilePath = targetDirectory + "customOO4.onnx";
	}
	else
	{
		targetFilePath = targetDirectory + "customOO1.onnx";
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
	emit appRunLog("训练启动....");

	if (_modelType == ModelType::BladeShape)
	{
		auto& global = GlobalStructData::getInstance();
		global.isTrainModel = true;
		emit updateTrainState(true);
		emit updateTrainTitle("开始训练刀型模型");

		emit appRunLog("清理旧的训练数据....");
		clear_older_trainData();

		//获取图片的label
		auto annotationGoodDataSet = annotation_data_set_bladeShape(false);
		auto annotationBadDataSet = annotation_data_set_bladeShape(true);

		auto dataSet = getDataSet(annotationGoodDataSet, 0);
		auto dataSetBad = getDataSet(annotationBadDataSet, 1);
		QString GoodSetLog = "其中正确的纽扣数据集有" + QString::number(dataSet.size()) + "条数据";
		QString BadSetLog = "其中错误的纽扣数据集有" + QString::number(dataSetBad.size()) + "条数据";
		emit appRunLog(GoodSetLog);
		emit appRunLog(BadSetLog);

		//拷贝训练数据
		emit appRunLog("拷贝训练文件");
		copyTrainData(dataSet);
		copyTrainData(dataSetBad);

		emit appRunLog("开始训练检测模型");
		trainShapeModel();

		exec();
	}
	else if (_modelType == ModelType::Color)
	{
		auto& global = GlobalStructData::getInstance();
		global.isTrainModel = true;
		emit updateTrainState(true);
		emit updateTrainTitle("开始训练色差模型");
		emit appRunLog("清理旧的训练数据....");

		auto annotationGoodDataSet1 = annotation_data_set_color(false, 1);
		auto annotationBadDataSet1 = annotation_data_set_color(true, 1);
		auto annotationGoodDataSet2 = annotation_data_set_color(false, 2);
		auto annotationBadDataSet2 = annotation_data_set_color(true, 2);
		auto annotationGoodDataSet3 = annotation_data_set_color(false, 3);
		auto annotationBadDataSet3 = annotation_data_set_color(true, 3);
		auto annotationGoodDataSet4 = annotation_data_set_color(false, 4);
		auto annotationBadDataSet4 = annotation_data_set_color(true, 4);

		dataSetGood1 = getDataSet(annotationGoodDataSet1, 0);
		dataSetGood2 = getDataSet(annotationGoodDataSet2, 0);
		dataSetGood3 = getDataSet(annotationGoodDataSet3, 0);
		dataSetGood4 = getDataSet(annotationGoodDataSet4, 0);

		dataSetBad1 = getDataSet(annotationBadDataSet1, 1);
		dataSetBad2 = getDataSet(annotationBadDataSet2, 1);
		dataSetBad3 = getDataSet(annotationBadDataSet3, 1);
		dataSetBad4 = getDataSet(annotationBadDataSet4, 1);

		emit appRunLog("清理旧的训练数据....");
		clear_older_trainData();
		emit appRunLog("拷贝训练文件");
		copyTrainData(dataSetGood1);
		copyTrainData(dataSetBad1);
		trainColorModel(1);
		exec();
	}

	quit();
}

QVector<AiTrainModule::labelAndImg> AiTrainModule::annotation_data_set_bladeShape(bool isBad)
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
		auto result = labelEngine->processImg(image);
		QString log = QString::number(i) + " ";

		auto processResultIndex = ImageProcessUtilty::getClassIndex(result);
		processResultIndex = ImageProcessUtilty::getAllIndexInMaxBody(result, processResultIndex, 10);
		if (processResultIndex[ClassId::Body].empty())
		{
			continue;
		}
		auto body = result[processResultIndex[ClassId::Body][0]];
		dataSet.emplaceBack(imagePath, body);
		log += "Area: " + QString::number(body.area) + " center_x" + QString::number(body.center_x) + " center_y" + QString::number(body.center_y);
		emit appRunLog(log);
		i++;
	}
	emit appRunLog("标注完" + QString::number(dataSet.size()) + "条数据");

	return dataSet;
}

QVector<AiTrainModule::labelAndImg> AiTrainModule::annotation_data_set_color(bool isBad, int workIndex)
{
	QVector<QString> imageList;
	if (isBad)
	{
		emit appRunLog("正在标注要筛选的纽扣数据集 工位:" + QString::number(workIndex));
		imageList = GlobalStructData::getInstance().modelStorageManager->getBadImagePathList(workIndex);
	}
	else
	{
		emit appRunLog("正在标注正确的纽扣的数据集 工位:" + QString::number(workIndex));
		imageList = GlobalStructData::getInstance().modelStorageManager->getGoodImagePathList(workIndex);
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
		auto result = labelEngine->processImg(image);
		QString log = QString::number(i) + " ";

		auto processResultIndex = ImageProcessUtilty::getClassIndex(result);
		processResultIndex = ImageProcessUtilty::getAllIndexInMaxBody(result, processResultIndex, 10);
		if (processResultIndex[ClassId::Body].empty())
		{
			continue;
		}
		auto body = result[processResultIndex[ClassId::Body][0]];
		dataSet.emplaceBack(imagePath, body);
		log += "Area: " + QString::number(body.area) + " center_x" + QString::number(body.center_x) + " center_y" + QString::number(body.center_y);
		emit appRunLog(log);
		i++;
	}
	emit appRunLog("标注完" + QString::number(dataSet.size()) + "条数据");

	return dataSet;
}

void AiTrainModule::exportModelToEngine()
{
	std::string batPath = R"(.\ConvertOnnxToEngine.bat)";
	_processExportToEngine->start("cmd.exe", { "/c", batPath.c_str() });
}

void AiTrainModule::exportColor1ModelToEngine()
{
	std::string batPath = R"(.\ConvertOnnxToEngineColor1.bat)";
	_processExportToEngineColor1->start("cmd.exe", { "/c", batPath.c_str() });
}

void AiTrainModule::exportColor2ModelToEngine()
{
	std::string batPath = R"(.\ConvertOnnxToEngineColor2.bat)";
	_processExportToEngineColor2->start("cmd.exe", { "/c", batPath.c_str() });
}

void AiTrainModule::exportColor3ModelToEngine()
{
	std::string batPath = R"(.\ConvertOnnxToEngineColor3.bat)";
	_processExportToEngineColor3->start("cmd.exe", { "/c", batPath.c_str() });
}

void AiTrainModule::exportColor4ModelToEngine()
{
	std::string batPath = R"(.\ConvertOnnxToEngineColor4.bat)";
	_processExportToEngineColor4->start("cmd.exe", { "/c", batPath.c_str() });
}

void AiTrainModule::handleProcessTrainModelBladeShapeOutput()
{
	QByteArray output = _processTrainModelBladeShape->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessTrainModelBladeShapeError()
{
	QByteArray errorOutput = _processTrainModelBladeShape->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
	int total = 100;
	int complete = -1;

	complete = parseProgressOO(errorStr, total);

	if (complete == -1)
	{
		return;
	}
	emit updateProgress(complete, 100);
}

void AiTrainModule::handleProcessTrainModelBladeShapeFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("正在导出模型");
		emit updateProgress(0, 0);
		exportModelToEngine();
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

void AiTrainModule::handleProcessExportEngineBladeShapeOutput()
{
	QByteArray output = _processExportToEngine->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessExportEngineBladeShapeError()
{
	QByteArray errorOutput = _processExportToEngine->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
}

void AiTrainModule::handleProcessExportEngineBladeShapeFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("导出完成");
		copyModelToTemp();
		packageModelToStorage();
		emit updateProgress(100, 100);
	}
	else
	{
		emit updateTrainTitle("导出失败");
	}
	auto& global = GlobalStructData::getInstance();
	global.isTrainModel = false;
	emit updateTrainState(false);
	quit();
}

void AiTrainModule::handleProcessTrainModelColor1Output()
{
	QByteArray output = _processTrainModelColor1->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessTrainModelColor1Error()
{
	QByteArray errorOutput = _processTrainModelColor1->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
	int total = 100;
	int complete = -1;

	complete = parseProgressOO(errorStr, total);

	if (complete == -1)
	{
		return;
	}
	complete = ((static_cast<double>(complete) / 100) * 25);
	emit updateProgress(complete, 100);
}

void AiTrainModule::handleProcessTrainModelColor1Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("正在导出模型1");
		emit updateProgress(0, 0);
		copyModelToTempColor(1);
		exportColor1ModelToEngine();
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

void AiTrainModule::handleProcessTrainModelColor2Error()
{
	QByteArray errorOutput = _processTrainModelColor2->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
	int total = 100;
	int complete = -1;

	complete = parseProgressOO(errorStr, total);

	if (complete == -1)
	{
		return;
	}
	complete = ((static_cast<double>(complete) / 100) * 25);
	emit updateProgress(complete + 25, 100);
}

void AiTrainModule::handleProcessTrainModelColor2Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("正在导出模型2");
		emit updateProgress(0, 0);
		copyModelToTempColor(2);
		exportColor2ModelToEngine();
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

void AiTrainModule::handleProcessTrainModelColor3Output()
{
	QByteArray output = _processTrainModelColor3->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessTrainModelColor3Error()
{
	QByteArray errorOutput = _processTrainModelColor3->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
	int total = 100;
	int complete = -1;

	complete = parseProgressOO(errorStr, total);

	if (complete == -1)
	{
		return;
	}
	complete = ((static_cast<double>(complete) / 100) * 25);
	emit updateProgress(complete + 50, 100);
}

void AiTrainModule::handleProcessTrainModelColor3Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("正在导出模型3");
		emit updateProgress(0, 0);
		copyModelToTempColor(3);
		exportColor3ModelToEngine();
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

void AiTrainModule::handleProcessTrainModelColor4Error()
{
	QByteArray errorOutput = _processTrainModelColor4->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
	int total = 100;
	int complete = -1;

	complete = parseProgressOO(errorStr, total);

	if (complete == -1)
	{
		return;
	}
	complete = ((static_cast<double>(complete) / 100) * 25);
	emit updateProgress(complete + 75, 100);
}

void AiTrainModule::handleProcessTrainModelColor4Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("正在导出模型4");
		emit updateProgress(0, 0);
		copyModelToTempColor(4);
		exportColor4ModelToEngine();
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

void AiTrainModule::handleProcessExportEngineColor1Error()
{
	QByteArray errorOutput = _processExportToEngineColor1->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
}

void AiTrainModule::handleProcessExportEngineColor1Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit appRunLog("清理旧的训练数据....");
		clear_older_trainData_color();
		emit appRunLog("拷贝训练文件");
		copyTrainData(dataSetGood2);
		copyTrainData(dataSetBad2);
		emit updateTrainTitle("正在训练");
		emit updateProgress(25, 100);
		trainColorModel(2);
	}
	else
	{
		emit updateTrainTitle("导出失败");
		emit updateTrainState(false);
		auto& global = GlobalStructData::getInstance();
		global.isTrainModel = false;
		quit();
	}
}

void AiTrainModule::handleProcessExportEngineColor2Error()
{
	QByteArray errorOutput = _processExportToEngineColor2->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
}

void AiTrainModule::handleProcessExportEngineColor2Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit appRunLog("清理旧的训练数据....");
		clear_older_trainData_color();
		emit appRunLog("拷贝训练文件");
		copyTrainData(dataSetGood3);
		copyTrainData(dataSetBad3);
		emit updateTrainTitle("正在训练");
		emit updateProgress(50, 100);
		trainColorModel(3);
	}
	else
	{
		emit updateTrainTitle("导出失败");
		emit updateTrainState(false);
		auto& global = GlobalStructData::getInstance();
		global.isTrainModel = false;
		quit();
	}
}

void AiTrainModule::handleProcessExportEngineColor3Error()
{
	QByteArray errorOutput = _processExportToEngineColor3->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
}

void AiTrainModule::handleProcessExportEngineColor3Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit appRunLog("清理旧的训练数据....");
		clear_older_trainData_color();
		emit appRunLog("拷贝训练文件");
		copyTrainData(dataSetGood4);
		copyTrainData(dataSetBad4);
		emit updateTrainTitle("正在训练");
		emit updateProgress(75, 100);
		trainColorModel(4);
	}
	else
	{
		emit updateTrainTitle("导出失败");
		emit updateTrainState(false);
		auto& global = GlobalStructData::getInstance();
		global.isTrainModel = false;
		quit();
	}
}

void AiTrainModule::handleProcessExportEngineColor4Error()
{
	QByteArray errorOutput = _processExportToEngineColor4->readAllStandardError();
	QString errorStr = QString::fromLocal8Bit(errorOutput);
	emit appRunLog(errorStr);
}

void AiTrainModule::handleProcessExportEngineColor4Finished(int exitCode, QProcess::ExitStatus exitStatus)
{
	if (exitStatus == QProcess::NormalExit)
	{
		emit updateTrainTitle("训练完成");
		emit updateProgress(100, 100);
		packageModelToStorage();
	}
	else
	{
		emit updateTrainTitle("导出失败");
	}
	auto& global = GlobalStructData::getInstance();
	global.isTrainModel = false;
	emit updateTrainState(false);
	quit();
}

void AiTrainModule::handleProcessExportEngineColor4Output()
{
	QByteArray output = _processExportToEngineColor4->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessExportEngineColor3Output()
{
	QByteArray output = _processExportToEngineColor3->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessExportEngineColor2Output()
{
	QByteArray output = _processExportToEngineColor2->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessExportEngineColor1Output()
{
	QByteArray output = _processExportToEngineColor1->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessTrainModelColor4Output()
{
	QByteArray output = _processTrainModelColor4->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::handleProcessTrainModelColor2Output()
{
	QByteArray output = _processTrainModelColor2->readAllStandardOutput();
	QString outputStr = QString::fromLocal8Bit(output);
	emit appRunLog(outputStr); // 将输出内容发送到日志或界面
}

void AiTrainModule::cancelTrain()
{
	emit updateTrainTitle("正在终止");
	if (_processTrainModelBladeShape->state() == QProcess::Running) {
		_processTrainModelBladeShape->kill();
		_processTrainModelBladeShape->waitForFinished();
	}
	if (_processExportToEngine->state() == QProcess::Running) {
		_processExportToEngine->kill();
		_processExportToEngine->waitForFinished();
	}
	if (_processTrainModelColor1->state() == QProcess::Running)
	{
		_processTrainModelColor1->kill();
		_processTrainModelColor1->waitForFinished();
	}
	if (_processTrainModelColor2->state() == QProcess::Running)
	{
		_processTrainModelColor2->kill();
		_processTrainModelColor2->waitForFinished();
	}
	if (_processTrainModelColor3->state() == QProcess::Running)
	{
		_processTrainModelColor3->kill();
		_processTrainModelColor3->waitForFinished();
	}
	if (_processTrainModelColor4->state() == QProcess::Running)
	{
		_processTrainModelColor4->kill();
		_processTrainModelColor4->waitForFinished();
	}
	if (_processExportToEngineColor1->state() == QProcess::Running)
	{
		_processExportToEngineColor1->kill();
		_processExportToEngineColor1->waitForFinished();
	}
	if (_processExportToEngineColor2->state() == QProcess::Running)
	{
		_processExportToEngineColor2->kill();
		_processExportToEngineColor2->waitForFinished();
	}
	if (_processExportToEngineColor3->state() == QProcess::Running)
	{
		_processExportToEngineColor3->kill();
		_processExportToEngineColor3->waitForFinished();
	}
	if (_processExportToEngineColor4->state() == QProcess::Running)
	{
		_processExportToEngineColor4->kill();
		_processExportToEngineColor4->waitForFinished();
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