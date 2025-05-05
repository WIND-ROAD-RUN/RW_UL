#include "stdafx.h"
#include "DlgModelManager.h"

#include"ModelStorageManager.h"
#include"GlobalStruct.h"
#include"ButtonUtilty.h"

DlgModelManager::DlgModelManager(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::DlgModelManagerClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
}

DlgModelManager::~DlgModelManager()
{
	delete ui;
}

void DlgModelManager::build_ui()
{
	_ModelListModel = new QStringListModel(this);
	ui->listView_modelList->setModel(_ModelListModel);

	_ModelInfoModel = new QStandardItemModel(this);
	ui->tableView_modelInfo->setModel(_ModelInfoModel);

	_loadingDialog = new LoadingDialog();
}

void DlgModelManager::build_connect()
{
	QObject::connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &DlgModelManager::pbtn_exit_clicked);
	QObject::connect(ui->listView_modelList->selectionModel(), &QItemSelectionModel::currentChanged,
		this, &DlgModelManager::onModelListSelectionChanged);
	QObject::connect(ui->pbtn_nextModel, &QPushButton::clicked,
		this, &DlgModelManager::pbtn_nextModel_clicked);
	QObject::connect(ui->pbtn_preModel, &QPushButton::clicked,
		this, &DlgModelManager::pbtn_preModel_clicked);
	QObject::connect(ui->pbtn_deleteModel, &QPushButton::clicked,
		this, &DlgModelManager::pbtn_deleteModel_clicked);
	QObject::connect(ui->pbtn_loadModel, &QPushButton::clicked,
		this, &DlgModelManager::pbtn_loadModel_clicked);
}

void DlgModelManager::copySOModel()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto currentIndex = ui->listView_modelList->currentIndex();
	if (!currentIndex.isValid()) {
		qDebug() << "未选择模型";
		return;
	}

	auto& config = _modelConfigs.at(currentIndex.row());
	QString sourceFile = QString::fromStdString(config.rootPath) + "/customSO.onnx";
	QString targetFile = globalPath.modelRootPath + "/customSO.onnx";

	// 检查源文件是否存在
	if (!QFile::exists(sourceFile)) {
		qDebug() << "源文件不存在:" << sourceFile;
		QMessageBox::warning(this, "错误", "源文件不存在: " + sourceFile);
		return;
	}

	// 如果目标文件已存在，则先删除
	if (QFile::exists(targetFile)) {
		if (!QFile::remove(targetFile)) {
			qDebug() << "无法删除目标文件:" << targetFile;
			QMessageBox::warning(this, "错误", "无法删除目标文件: " + targetFile);
			return;
		}
	}

	// 拷贝文件
	if (QFile::copy(sourceFile, targetFile)) {
		globalStruct.imageProcessingModule1->reloadSOModel();
		globalStruct.imageProcessingModule2->reloadSOModel();
		globalStruct.imageProcessingModule3->reloadSOModel();
		globalStruct.imageProcessingModule4->reloadSOModel();
		qDebug() << "文件拷贝成功:" << sourceFile << "到" << targetFile;
		auto& globalStruct = GlobalStructData::getInstance();
		globalStruct.isOpenColor = true;
	}
	else {
		qDebug() << "文件拷贝失败:" << sourceFile << "到" << targetFile;
		QMessageBox::warning(this, "错误", "文件拷贝失败: " + sourceFile + " 到 " + targetFile);
	}
}

void DlgModelManager::pbtn_exit_clicked()
{
	this->hide();
}

void DlgModelManager::onModelListSelectionChanged(const QModelIndex& current, const QModelIndex& previous)

{
	// 检查当前索引是否有效
	if (!current.isValid()) {
		return;
	}

	// 获取当前选择的模型索引
	int selectedIndex = current.row();

	// 刷新模型信息表和示例图片
	flashModelInfoTable(selectedIndex);
	flashExampleImage(selectedIndex);
}

void DlgModelManager::pbtn_nextModel_clicked()
{
	auto maxIndex = _modelConfigs.size();
	auto currentIndex = ui->listView_modelList->currentIndex();
	ui->listView_modelList->setCurrentIndex(currentIndex.siblingAtColumn(0).siblingAtRow((currentIndex.row() + 1) % maxIndex));
}

void DlgModelManager::pbtn_preModel_clicked()
{
	auto maxIndex = _modelConfigs.size();
	auto currentIndex = ui->listView_modelList->currentIndex();
	ui->listView_modelList->setCurrentIndex(currentIndex.siblingAtColumn(0).siblingAtRow((currentIndex.row() - 1 + maxIndex) % maxIndex));
}

void DlgModelManager::pbtn_deleteModel_clicked()
{
	auto isDelete = QMessageBox::question(this, "确定", "你真的要删除吗？该模型所有数据将会被清空");
	if (isDelete != QMessageBox::Yes)
	{
		return;
	}
	auto currentIndex = ui->listView_modelList->currentIndex();
	if (!currentIndex.isValid()) {
		return;
	}

	auto& targetPath = _configIndex.modelIndexs.at(currentIndex.row()).root_path;
	deleteDirectory(QString::fromStdString(targetPath));

	_ModelListModel->removeRows(currentIndex.row(), 1);
	auto deleteConfig = _configIndex.modelIndexs.at(currentIndex.row());
	_configIndex.deleteConfig(deleteConfig);
	auto& globalStruct = GlobalStructData::getInstance();
	auto& modelStorageManager = globalStruct.modelStorageManager;

	rw::cdm::AiModelConfig configItem = _modelConfigs.at(currentIndex.row());
	modelStorageManager->eraseModelConfig(configItem);
	_modelConfigs.remove(currentIndex.row());

	// 清空模型信息表和示例图片（可选）
	flashModelInfoTable(0);
	flashExampleImage(0);
}

void DlgModelManager::pbtn_loadModel_clicked()
{
	auto& globalStruct = GlobalStructData::getInstance();
	globalStruct.isOpenBladeShape = false;
	globalStruct.isOpenColor = false;
	_loadingDialog->show();
	_loadingDialog->updateMessage("加载中");
	auto currentIndex = ui->listView_modelList->currentIndex();
	auto& config = _modelConfigs.at(currentIndex.row());
	if (config.modelType == rw::cdm::ModelType::BladeShape)
	{
		copyOOModel();
		globalStruct.isOpenBladeShape = true;
		globalStruct.isOpenColor = false;
	}
	else if (config.modelType == rw::cdm::ModelType::Color)
	{
		copySOModel();
		globalStruct.isOpenBladeShape = false;
		globalStruct.isOpenColor = true;
	}
	else
	{
		QMessageBox::warning(this, "警告", "模型类型不支持");
		_loadingDialog->hide();
		return;
	}

	if (globalStruct.camera1)
	{
		globalStruct.camera1->setExposureTime(config.exposureTime);
		globalStruct.camera1->setGain(config.gain);
	}
	if (globalStruct.camera2)
	{
		globalStruct.camera2->setExposureTime(config.exposureTime);
		globalStruct.camera2->setGain(config.gain);
	}
	if (globalStruct.camera3)
	{
		globalStruct.camera3->setExposureTime(config.exposureTime);
		globalStruct.camera3->setGain(config.gain);
	}
	if (globalStruct.camera4)
	{
		globalStruct.camera4->setExposureTime(config.exposureTime);
		globalStruct.camera4->setGain(config.gain);
	}

	if (config.upLight)
	{
		globalStruct.setUpLight(true);
	}
	else
	{
		globalStruct.setUpLight(false);
	}

	if (config.downLight)
	{
		globalStruct.setDownLight(true);
	}
	else
	{
		globalStruct.setDownLight(false);
	}

	if (config.sideLight)
	{
		globalStruct.setSideLight(true);
	}
	else
	{
		globalStruct.setSideLight(false);
	}

	copyTargetImageFromStorageInTemp();
	_loadingDialog->hide();
	this->hide();
}

void DlgModelManager::showEvent(QShowEvent* show_event)
{
	QDialog::showEvent(show_event);
	flashModelList();
	flashModelInfoTable(0);
	flashExampleImage(0);
}

QString DlgModelManager::formatDateString(const std::string& dateStr)
{
	// 将 std::string 转换为 QString
	QString dateString = QString::fromStdString(dateStr);

	// 使用 QDateTime 解析日期字符串
	QDateTime dateTime = QDateTime::fromString(dateString, "yyyyMMddHHmmss");

	// 检查解析是否成功
	if (dateTime.isValid()) {
		// 格式化为 "2025年4月22日6时7分1秒" 的形式
		return dateTime.toString("yyyy年M月d日H时m分s秒");
	}
	else {
		// 如果解析失败，返回提示信息
		return QString::fromStdString(dateStr);
	}
}

QVector<QString> DlgModelManager::getImagePaths(const QString& rootPath, bool isGood)
{
	QVector<QString> imagePaths;
	QString rootImagePath = rootPath + "/Image/";
	QDir rootDir(rootImagePath);

	if (!rootDir.exists()) {
		qDebug() << "根路径不存在:" << rootImagePath;
		return imagePaths;
	}

	// 定义目标文件夹名称
	QString targetFolder = isGood ? "good" : "bad";

	// 遍历 work1, work2, work3, work4 文件夹
	QStringList workFolders = rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& workFolder : workFolders) {
		QDir workDir(rootDir.filePath(workFolder));
		QDir targetDir(workDir.filePath(targetFolder));

		if (!targetDir.exists()) {
			qDebug() << "目标文件夹不存在:" << targetDir.absolutePath();
			continue;
		}

		// 查找目标文件夹下的所有 .png 文件
		QStringList filters;
		filters << "*.png";
		QStringList pngFiles = targetDir.entryList(filters, QDir::Files | QDir::NoSymLinks);

		// 获取每个文件的绝对路径并添加到结果中
		for (const QString& pngFile : pngFiles) {
			imagePaths.append(targetDir.absoluteFilePath(pngFile));
		}
	}

	return imagePaths;
}

QVector<QString> DlgModelManager::getImagePaths(const QString& rootPath, bool isGood, int maxCount)
{
	QVector<QString> imagePaths;
	QString rootImagePath = rootPath + "/Image/";
	QDir rootDir(rootImagePath);

	if (!rootDir.exists()) {
		qDebug() << "根路径不存在:" << rootImagePath;
		return imagePaths;
	}

	// 定义目标文件夹名称
	QString targetFolder = isGood ? "good" : "bad";

	// 遍历 work1, work2, work3, work4 文件夹
	QStringList workFolders = rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& workFolder : workFolders) {
		QDir workDir(rootDir.filePath(workFolder));
		QDir targetDir(workDir.filePath(targetFolder));

		if (!targetDir.exists()) {
			qDebug() << "目标文件夹不存在:" << targetDir.absolutePath();
			continue;
		}

		// 查找目标文件夹下的所有 .png 文件
		QStringList filters;
		filters << "*.png";
		QStringList pngFiles = targetDir.entryList(filters, QDir::Files | QDir::NoSymLinks);

		// 获取每个文件的绝对路径并添加到结果中
		for (const QString& pngFile : pngFiles) {
			imagePaths.append(targetDir.absoluteFilePath(pngFile));

			// 如果已找到指定数量的图片，停止搜索
			if (imagePaths.size() >= maxCount) {
				return imagePaths;
			}
		}
	}

	return imagePaths;
}

void DlgModelManager::deleteDirectory(const QString& targetPath)
{
	QDir dir(targetPath);

	// 检查路径是否存在
	if (!dir.exists()) {
		qDebug() << "路径不存在:" << targetPath;
		return;
	}

	// 使用 QDir 的 removeRecursively 方法删除路径及其内容
	if (dir.removeRecursively()) {
		qDebug() << "成功删除路径及其内容:" << targetPath;
	}
	else {
		qDebug() << "删除路径失败:" << targetPath;
	}
}

QString DlgModelManager::findXmlFile(const QString& rootPath)
{
	// 检查路径是否有效
	QDir dir(rootPath);
	if (!dir.exists()) {
		qDebug() << "路径不存在:" << rootPath;
		return QString(); // 返回空路径
	}

	// 设置过滤器，只查找 .xml 文件
	QStringList filters;
	filters << "*.xml";

	// 获取所有匹配的文件
	QStringList xmlFiles = dir.entryList(filters, QDir::Files | QDir::NoSymLinks);

	// 如果找到文件，返回第一个文件的绝对路径
	if (!xmlFiles.isEmpty()) {
		QString absolutePath = dir.absoluteFilePath(xmlFiles.first());
		return absolutePath;
	}

	return QString();
}

void DlgModelManager::flashModelList()
{
	_modelConfigs.clear();
	auto& globalStruct = GlobalStructData::getInstance();
	auto& modelStorageManager = globalStruct.modelStorageManager;
	_configIndex = modelStorageManager->getModelConfigIndex();

	QStringList data;
	for (const auto& item : _configIndex.modelIndexs)
	{
		QString modelTypeStr = QString::fromStdString(
			item.model_type == rw::cdm::ModelType::Undefined ? "未定义" :
			item.model_type == rw::cdm::ModelType::BladeShape ? "刀型" :
			item.model_type == rw::cdm::ModelType::Color ? "颜色" : "未知");

		data << QString::fromStdString(item.model_name) + " (" + modelTypeStr + ")";

		auto configPath = findXmlFile(QString::fromStdString(item.root_path));
		if (configPath.isEmpty())
		{
			QMessageBox::warning(this, "警告", QString::fromStdString(item.root_path) + "模型配置文件不存在");
			rw::cdm::AiModelConfig aiModelConfig;
			_modelConfigs.push_back(aiModelConfig);
			continue;
		}
		auto config = globalStruct.storeContext->load(configPath.toStdString());
		rw::cdm::AiModelConfig aiModelConfig(*config);
		_modelConfigs.push_back(aiModelConfig);
	}

	_ModelListModel->setStringList(data);

	if (!_ModelListModel->stringList().isEmpty())
	{
		QModelIndex firstIndex = _ModelListModel->index(0, 0); // 获取第一个项的索引
		ui->listView_modelList->setCurrentIndex(firstIndex);   // 设置为当前选择项
	}
}

void DlgModelManager::flashModelInfoTable(size_t index)
{
	_ModelInfoModel->clear();
	// 原始的列标题
	QStringList originalHeaders = QStringList() << "模型名称:" << "ID:" << "模型类型:" << "上光源:"
		<< "侧光源:" << "下光源:" << "曝光:"
		<< "增益:" << "模型根路径:" << "训练日期:";

	// 设置旋转后的行标题（原始列标题变为行标题）
	_ModelInfoModel->setVerticalHeaderLabels(originalHeaders);

	// 设置第一列无标题
	_ModelInfoModel->setHeaderData(0, Qt::Horizontal, QVariant()); // 清空第一列的标题

	if (_configIndex.modelIndexs.size() <= index)
	{
		return;
	}
	rw::cdm::AiModelConfig aiModelConfig = _modelConfigs.at(index);

	QStandardItem* nameItem = new QStandardItem();
	nameItem->setText(QString::fromStdString(aiModelConfig.name));
	_ModelInfoModel->setItem(0, 0, nameItem);

	QStandardItem* idItem = new QStandardItem();
	idItem->setText(QString::number(aiModelConfig.id));
	_ModelInfoModel->setItem(1, 0, idItem);

	QStandardItem* modelTypeItem = new QStandardItem();
	modelTypeItem->setText(QString::fromStdString(
		aiModelConfig.modelType == rw::cdm::ModelType::Undefined ? "未定义" :
		aiModelConfig.modelType == rw::cdm::ModelType::BladeShape ? "刀型" :
		aiModelConfig.modelType == rw::cdm::ModelType::Color ? "颜色" : "未知"));
	_ModelInfoModel->setItem(2, 0, modelTypeItem);

	QStandardItem* upLightItem = new QStandardItem();
	upLightItem->setText(aiModelConfig.upLight ? "开启" : "关闭");
	_ModelInfoModel->setItem(3, 0, upLightItem);

	QStandardItem* sideLightItem = new QStandardItem();
	sideLightItem->setText(aiModelConfig.sideLight ? "开启" : "关闭");
	_ModelInfoModel->setItem(4, 0, sideLightItem);

	QStandardItem* downLightItem = new QStandardItem();
	downLightItem->setText(aiModelConfig.downLight ? "开启" : "关闭");
	_ModelInfoModel->setItem(5, 0, downLightItem);

	QStandardItem* exposureItem = new QStandardItem();
	exposureItem->setText(QString::number(aiModelConfig.exposureTime));
	_ModelInfoModel->setItem(6, 0, exposureItem);

	QStandardItem* gainItem = new QStandardItem();
	gainItem->setText(QString::number(aiModelConfig.gain));
	_ModelInfoModel->setItem(7, 0, gainItem);

	QStandardItem* rootPathItem = new QStandardItem();
	rootPathItem->setText(QString::fromStdString(aiModelConfig.rootPath));
	_ModelInfoModel->setItem(8, 0, rootPathItem);

	QStandardItem* dateItem = new QStandardItem();
	dateItem->setText(formatDateString(aiModelConfig.date));
	_ModelInfoModel->setItem(9, 0, dateItem);

	ui->tableView_modelInfo->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
	ui->tableView_modelInfo->horizontalHeader()->hide();
}

void DlgModelManager::flashExampleImage(size_t index)
{
	if (_configIndex.modelIndexs.size() <= index)
	{
		return;
	}
	auto& config = _modelConfigs.at(index);
	auto& rootPath = config.rootPath;
	auto goodImageList = getImagePaths(QString::fromStdString(rootPath), true, 1);
	auto badImageList = getImagePaths(QString::fromStdString(rootPath), false, 1);

	if (goodImageList.size() == 0)
	{
		ui->label_imgDisplayOK->setText("未找到图片");
	}
	else
	{
		QPixmap image(goodImageList.at(0));
		ui->label_imgDisplayOK->setPixmap(image.scaled(ui->label_imgDisplayOK->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}

	if (badImageList.size() == 0)
	{
		ui->label_imgDisplayNG->setText("未找到图片");
	}
	else
	{
		QPixmap image(badImageList.at(0));
		ui->label_imgDisplayNG->setPixmap(image.scaled(ui->label_imgDisplayNG->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}
}

void DlgModelManager::copyTargetImageFromStorageInTemp()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto& modelStorageManager = globalStruct.modelStorageManager;
	//清理临时目录
	modelStorageManager->clear_temp();
	auto currentIndex = ui->listView_modelList->currentIndex();
	if (!currentIndex.isValid()) {
		return;
	}

	auto& config = _modelConfigs.at(currentIndex.row());
	auto rootPath = config.rootPath;

	auto& tempRootPath = globalPath.modelStorageManagerTempPath;

	QString sourceImagePath = QString::fromStdString(rootPath + "/Image");
	QString targetImagePath = tempRootPath + "/Image";

	QDir sourceDir(sourceImagePath);
	if (!sourceDir.exists()) {
		qDebug() << "源路径不存在:" << sourceImagePath;
		return;
	}

	QDir targetDir(targetImagePath);
	if (!targetDir.exists()) {
		if (!targetDir.mkpath(targetImagePath)) {
			qDebug() << "无法创建目标路径:" << targetImagePath;
			return;
		}
	}

	// 获取源路径下的所有文件和子目录
	QStringList files = sourceDir.entryList(QDir::Files);
	for (const QString& fileName : files) {
		QString sourceFile = sourceDir.filePath(fileName);
		QString targetFile = targetDir.filePath(fileName);

		// 如果目标文件已存在，则覆盖
		if (QFile::exists(targetFile)) {
			QFile::remove(targetFile);
		}

		if (!QFile::copy(sourceFile, targetFile)) {
			qDebug() << "无法拷贝文件:" << sourceFile << "到" << targetFile;
		}
	}

	// 递归拷贝子目录
	QStringList directories = sourceDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& dirName : directories) {
		QString sourceSubDir = sourceDir.filePath(dirName);
		QString targetSubDir = targetDir.filePath(dirName);

		copyDirectoryRecursively(sourceSubDir, targetSubDir);
	}
}

void DlgModelManager::copyOOModel()
{
	auto& globalStruct = GlobalStructData::getInstance();
	auto currentIndex = ui->listView_modelList->currentIndex();
	if (!currentIndex.isValid()) {
		qDebug() << "未选择模型";
		return;
	}

	auto& config = _modelConfigs.at(currentIndex.row());
	QString sourceFile = QString::fromStdString(config.rootPath) + "/customOO.onnx";
	QString targetFile = globalPath.modelRootPath + "/customOO.onnx";

	// 检查源文件是否存在
	if (!QFile::exists(sourceFile)) {
		qDebug() << "源文件不存在:" << sourceFile;
		QMessageBox::warning(this, "错误", "源文件不存在: " + sourceFile);
		return;
	}

	// 如果目标文件已存在，则先删除
	if (QFile::exists(targetFile)) {
		if (!QFile::remove(targetFile)) {
			qDebug() << "无法删除目标文件:" << targetFile;
			QMessageBox::warning(this, "错误", "无法删除目标文件: " + targetFile);
			return;
		}
	}

	// 拷贝文件
	if (QFile::copy(sourceFile, targetFile)) {
		globalStruct.imageProcessingModule1->reloadOOModel();
		globalStruct.imageProcessingModule2->reloadOOModel();
		globalStruct.imageProcessingModule3->reloadOOModel();
		globalStruct.imageProcessingModule4->reloadOOModel();
		qDebug() << "文件拷贝成功:" << sourceFile << "到" << targetFile;
		auto& globalStruct = GlobalStructData::getInstance();
		globalStruct.isOpenBladeShape = false;
	}
	else {
		qDebug() << "文件拷贝失败:" << sourceFile << "到" << targetFile;
		QMessageBox::warning(this, "错误", "文件拷贝失败: " + sourceFile + " 到 " + targetFile);
	}
}

bool DlgModelManager::copyDirectoryRecursively(const QString& sourceDirPath, const QString& targetDirPath)
{
	QDir sourceDir(sourceDirPath);
	if (!sourceDir.exists()) {
		return false;
	}

	QDir targetDir(targetDirPath);
	if (!targetDir.exists()) {
		if (!targetDir.mkpath(targetDirPath)) {
			return false;
		}
	}

	// 拷贝文件
	QStringList files = sourceDir.entryList(QDir::Files);
	for (const QString& fileName : files) {
		QString sourceFile = sourceDir.filePath(fileName);
		QString targetFile = targetDir.filePath(fileName);

		// 如果目标文件已存在，则覆盖
		if (QFile::exists(targetFile)) {
			QFile::remove(targetFile);
		}

		if (!QFile::copy(sourceFile, targetFile)) {
			return false;
		}
	}

	// 递归拷贝子目录
	QStringList directories = sourceDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& dirName : directories) {
		QString sourceSubDir = sourceDir.filePath(dirName);
		QString targetSubDir = targetDir.filePath(dirName);

		if (!copyDirectoryRecursively(sourceSubDir, targetSubDir)) {
			return false;
		}
	}

	return true;
}