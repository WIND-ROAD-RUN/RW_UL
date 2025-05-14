#include "ModelConverter.h"

#include <qtabbar.h>

#include<QFileDialog>
#include<QMessageBox>

ModelConverter::ModelConverter(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::ModelConverterClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
	_converter = new Converter(this);
	connect(_converter, &Converter::appRunLog,
		this, &ModelConverter::on_appRunLog);
	connect(_converter, &Converter::finish,
		this, &ModelConverter::finish);
}

ModelConverter::~ModelConverter()
{
	isKill = true;
	if (_converter)
	{
		_converter->cancel();
		delete _converter;
	}
	delete ui;
}

void ModelConverter::build_ui()
{
	ui->tabWidget->setCurrentIndex(0);
	ui->tabWidget->tabBar()->hide();
}

void ModelConverter::build_connect()
{
	connect(ui->pbtn_openOnnxFilePath, &QPushButton::clicked,
		this, &ModelConverter::pbtn_openOnnxFilePath_clicked);
	connect(ui->pbtn_outPutPath, &QPushButton::clicked,
		this, &ModelConverter::pbtn_outPutPath_clicked);
	connect(ui->pbtn_exit, &QPushButton::clicked, 
		this, &ModelConverter::pbtn_exit_clicked);
	connect(ui->pbtn_nextStep, &QPushButton::clicked, 
		this, &ModelConverter::pbtn_nextStep);
	connect(ui->pbtn_tab1Exit, &QPushButton::clicked,
		this, &ModelConverter::pbtn_tab1Exit_clicked);
	connect(ui->pbtn_preStep, &QPushButton::clicked,
		this, &ModelConverter::pbtn_preStep_clicked);
	connect(ui->pbtn_startConvert, &QPushButton::clicked,
		this, &ModelConverter::pbtn_startConvert_clicked);

}

void ModelConverter::pbtn_openOnnxFilePath_clicked()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open ONNX File"), "", tr("ONNX Files (*.onnx)"));
	if (!fileName.isEmpty())
	{
		ui->lEdit_onnxFilePath->setText(fileName);

		QString dir = QFileInfo(fileName).absolutePath();
		ui->lEdit_outPutPath->setText(dir);
	}
}

void ModelConverter::pbtn_outPutPath_clicked()
{
	QString dir = QFileDialog::getExistingDirectory(this, tr("Select Output Directory"), "", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
	if (!dir.isEmpty())
	{
		ui->lEdit_outPutPath->setText(dir);
	}
}

void ModelConverter::pbtn_exit_clicked()
{
	this->close();
}

void ModelConverter::pbtn_nextStep()
{
	if (ui->lEdit_onnxFilePath->text().isEmpty())
	{
		QMessageBox::warning(this, tr("Warning"), tr("Please select an ONNX file."));
		return;
	}
	if (ui->lEdit_outPutPath->text().isEmpty())
	{
		QMessageBox::warning(this, tr("Warning"), tr("Please select an output directory."));
		return;
	}
	ui->tabWidget->setCurrentIndex(1);
	ui->progressBar->setRange(0, 100);
	ui->progressBar->setValue(0);
}

void ModelConverter::pbtn_tab1Exit_clicked()
{
	this->close();
}

void ModelConverter::pbtn_preStep_clicked()
{
	ui->tabWidget->setCurrentIndex(0);
}

void ModelConverter::pbtn_startConvert_clicked()
{
	ui->progressBar->setRange(0, 0);
	auto inputFile = ui->lEdit_onnxFilePath->text();
	auto outPutPath = ui->lEdit_outPutPath->text();
	auto outPutName = outPutPath+R"(\)"+ R"(best.engine)";
	_converter->inputFile = inputFile;
	_converter->outputFile = outPutName;
	ui->pbtn_startConvert->setEnabled(false);
	ui->pbtn_tab1Exit->setEnabled(false);
	ui->pbtn_preStep->setEnabled(false);
	_converter->run();
}

void ModelConverter::on_appRunLog(QString log)
{
	ui->plainTextEdit->appendPlainText(log);
}

void ModelConverter::finish()
{
	ui->progressBar->setRange(0, 100);
	ui->progressBar->setValue(100);
	ui->pbtn_startConvert->setEnabled(true);
	ui->pbtn_tab1Exit->setEnabled(true);
	ui->pbtn_preStep->setEnabled(true);
	ui->plainTextEdit->appendPlainText("Convert finished.");
	if (!isKill)
	{
		QMessageBox::information(this, tr("Info"), tr("Convert finished."));
	}
}
