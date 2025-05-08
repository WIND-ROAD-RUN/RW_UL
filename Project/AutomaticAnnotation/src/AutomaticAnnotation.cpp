#include "AutomaticAnnotation.h"

#include"NumberKeyboard.h"


#include"rqw_ImagePainter.h"

#include<QMessageBox>
#include<QFileDialog>
#include<QDirIterator>

static QStringList getAllImagePaths(const QString & path)
{
	QString folderPath = path; // 获取目录路径
	if (folderPath.isEmpty()) {
		return QStringList();
	}

	QDir dir(folderPath);
	if (!dir.exists()) {
		return QStringList();
	}

	// 定义支持的图片格式
	QStringList imageFilters = { "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp" };
	QStringList imagePaths;

	// 使用 QDirIterator 递归遍历目录
	QDirIterator it(folderPath, imageFilters, QDir::Files, QDirIterator::Subdirectories);
	while (it.hasNext()) {
		imagePaths.append(it.next()); // 获取图片的绝对路径
	}

	return imagePaths;
}

AutomaticAnnotation::AutomaticAnnotation(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::AutomaticAnnotationClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
}

AutomaticAnnotation::~AutomaticAnnotation()
{
	for (int i = 0;i < threads.size();i++)
	{
		disconnect(threads[i], &AutomaticAnnotationThread::imageProcessed, this, &AutomaticAnnotation::displayImage);
	}

	for (int i = 0; i < threads.size(); i++) {
		threads[i]->wait();
		delete threads[i];
	}
	delete ui;
}

void AutomaticAnnotation::build_ui()
{
	ui->tabWidget->setCurrentIndex(0);
	viewer = new PicturesViewer(this);

	ui->cBox_checkDeployType->addItem("OnnxRuntime");
	ui->cBox_checkDeployType->addItem("TensorRT");
	ui->cBox_checkDeployType->setCurrentIndex(0);

	ui->cBox_checkModelType->addItem("Yolov11_seg");
	ui->cBox_checkModelType->addItem("Yolov11_obb");
	ui->cBox_checkModelType->setCurrentIndex(0);
}

void AutomaticAnnotation::build_connect()
{
	QObject::connect(ui->pbtn_setImageInput, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_setImageInput_clicked);
	QObject::connect(ui->pbtn_setLabelOutput, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_setLabelOutput_clicked);
	QObject::connect(ui->pbtn_setImageOutput, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_setImageOutput_clicked);
	QObject::connect(ui->pbtn_setModelPath, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_setModelPath_clicked);
	QObject::connect(ui->pbtn_setWorkers, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_setWorkers_clicked);
	QObject::connect(ui->pbtn_setConfThreshold, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_setConfThreshold_clicked);
	QObject::connect(ui->pbtn_nmsThreshold, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_nmsThreshold_clicked);
	QObject::connect(ui->pbtn_exit, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_exit_clicked);
	QObject::connect(ui->pbtn_LookImage, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_LookImage_clicked);
	QObject::connect(ui->pbtn_next, &QPushButton::clicked
		, this, &AutomaticAnnotation::pbtn_next_clicked);
}

void AutomaticAnnotation::pbtn_setImageInput_clicked()
{
	QString folderPath = QFileDialog::getExistingDirectory(this, tr("Select Folder"), "");
	if (!folderPath.isEmpty())
	{
		ui->lineEdit_ImageInput->setText(folderPath);
	}
	else
	{
		ui->lineEdit_ImageInput->clear();
	}
}

void AutomaticAnnotation::pbtn_setLabelOutput_clicked()
{
	QString folderPath = QFileDialog::getExistingDirectory(this, tr("Select Folder"), "");
	if (!folderPath.isEmpty())
	{
		ui->lineEdit_labelOutput->setText(folderPath);
	}
	else
	{
		ui->lineEdit_labelOutput->clear();
	}
}

void AutomaticAnnotation::pbtn_setImageOutput_clicked()
{
	QString folderPath = QFileDialog::getExistingDirectory(this, tr("Select Folder"), "");
	if (!folderPath.isEmpty())
	{
		ui->lineEdit_ImageOutput->setText(folderPath);
	}
	else
	{
		ui->lineEdit_ImageOutput->clear();
	}
}

void AutomaticAnnotation::pbtn_setModelPath_clicked()
{
	QString filePath = QFileDialog::getOpenFileName(this, tr("Select File"), "", tr("Model Files (*.onnx *.pb *.xml *.engine)"));
	if (!filePath.isEmpty())
	{
		ui->lineEdit_modelPath->setText(filePath);
	}
	else
	{
		ui->lineEdit_modelPath->clear();
	}
}

void AutomaticAnnotation::pbtn_setWorkers_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		ui->pbtn_setWorkers->setText(value);
	}
}

void AutomaticAnnotation::pbtn_setConfThreshold_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		ui->pbtn_setConfThreshold->setText(value);
	}
}

void AutomaticAnnotation::pbtn_nmsThreshold_clicked()
{
	NumberKeyboard numKeyBord;
	numKeyBord.setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
	auto isAccept = numKeyBord.exec();
	if (isAccept == QDialog::Accepted)
	{
		auto value = numKeyBord.getValue();
		ui->pbtn_nmsThreshold->setText(value);
	}
}

void AutomaticAnnotation::pbtn_exit_clicked()
{
	this->close();
}

void AutomaticAnnotation::pbtn_LookImage_clicked()
{
	viewer->setRootPath(ui->lineEdit_ImageInput->text());
	viewer->show();
}

void AutomaticAnnotation::pbtn_next_clicked()
{
	ui->tabWidget->setCurrentIndex(1);
	auto currentDeployType = ui->cBox_checkDeployType->currentText();
	auto currentModelType = ui->cBox_checkModelType->currentText();
	auto imageInput = ui->lineEdit_ImageInput->text();
	auto labelOutput = ui->lineEdit_labelOutput->text();
	auto imageOutput = ui->lineEdit_ImageOutput->text();
	auto modelPath = ui->lineEdit_modelPath->text();
	auto workers = ui->pbtn_setWorkers->text().toInt();
	auto confThreshold = ui->pbtn_setConfThreshold->text();
	auto nmsThreshold = ui->pbtn_nmsThreshold->text();
	rw::ModelEngineConfig config;
	config.modelPath = modelPath.toStdString();
	config.nms_threshold = std::stof(nmsThreshold.toStdString());
	config.conf_threshold = std::stof(confThreshold.toStdString());

	auto paths = getAllImagePaths(ui->lineEdit_ImageInput->text());

	int averageNum = paths.size() / workers;
	int lastNum = paths.size() % workers;
	for (int i = 0; i < workers; i++) {
		AutomaticAnnotationThread* thread = nullptr;
		if (i == 0) {
			thread = new AutomaticAnnotationThread(paths.mid(i, averageNum + lastNum));
		}
		else {
			thread = new AutomaticAnnotationThread(paths.mid(i * averageNum, averageNum));
		}
		thread->config = config;
		threads.push_back(thread);
		connect(thread, &AutomaticAnnotationThread::imageProcessed, this, &AutomaticAnnotation::displayImage, Qt::QueuedConnection);
	}
}

void AutomaticAnnotation::on_pbtn_preStep_clicked()
{
	for (int i = 0;i < threads.size();i++)
	{
		disconnect(threads[i], &AutomaticAnnotationThread::imageProcessed, this, &AutomaticAnnotation::displayImage);
	}

	for (int i = 0; i < threads.size(); i++) {
		threads[i]->wait();
		delete threads[i];
	}
	threads.clear();
	ui->tabWidget->setCurrentIndex(0);
}

void AutomaticAnnotation::on_pbtn_startAnnotation_clicked()
{
	for (int i = 0; i < threads.size(); i++) {
		threads[i]->start();
	}
}

void AutomaticAnnotation::on_pbtn_tab2_exit_clicked()
{
	this->close();
}

void AutomaticAnnotation::displayImage(QString imagePath, QPixmap pixmap)
{
	if (pixmap.isNull()) {
		qDebug() << "Failed to load image:" << imagePath;
		return;
	}
	ui->label_imgDisplay->setPixmap(pixmap.scaled(ui->label_imgDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

