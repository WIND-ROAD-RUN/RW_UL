#include "PictureViewerThumbnails.h"

#include "ui_PictureViewerThumbnails.h"
#include <QVBoxLayout>
#include <QScrollBar>
#include <QFileInfo>
#include <QImageReader>
#include <QPixmap>
#include <QIcon>
#include <QPainter>
#include <QDir>
#include <QMessageBox>

PictureViewerThumbnails::PictureViewerThumbnails(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::PictureViewerThumbnailsClass())
{
	ui->setupUi(this);

	build_ui();
	build_connect();
}

PictureViewerThumbnails::~PictureViewerThumbnails()
{
	delete ui;
}

void PictureViewerThumbnails::setRootPath(const QString& rootPath)
{
	m_rootPath = rootPath;
}

void PictureViewerThumbnails::setSize(const QSize& size)
{
	m_thumbnailSize = size;
	_listWidget->setIconSize(m_thumbnailSize);

	QSize cellSize(m_thumbnailSize.width() + 16, m_thumbnailSize.height() + 32);
	for (int i = 0; i < _listWidget->count(); ++i) {
		QListWidgetItem* item = _listWidget->item(i);
		item->setSizeHint(cellSize);
		loadThumbnail(item->data(Qt::UserRole).toString(), item);
	}
}

void PictureViewerThumbnails::showEvent(QShowEvent* event)
{
	_loadingDialog->updateMessage("加载图片中");
	_loadingDialog->show();
	loadImageList();
	_loadingDialog->close();
	QMainWindow::showEvent(event);
}

void PictureViewerThumbnails::build_ui()
{
	_listWidget = new DraggableListWidget(this);
	auto listWidgetLayout = new QVBoxLayout(this);
	listWidgetLayout->addWidget(_listWidget);
	ui->groupBox->setLayout(listWidgetLayout);

	_listWidget->setViewMode(QListView::IconMode);
	_listWidget->setResizeMode(QListView::Adjust);
	_listWidget->setIconSize(m_thumbnailSize);
	_listWidget->setMovement(QListView::Static);
	_listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
	_listWidget->setSpacing(8);

	_loadingDialog = new LoadingDialog();
}

void PictureViewerThumbnails::build_connect()
{
	connect(_listWidget, &QListWidget::itemSelectionChanged, 
		this, &PictureViewerThumbnails::onThumbnailSelected);

	connect(ui->pbtn_delete, &QPushButton::clicked,
		this, &PictureViewerThumbnails::pbtn_delete_clicked);
	connect(ui->pbtn_deleteTotal, &QPushButton::clicked,
		this, &PictureViewerThumbnails::pbtn_deleteTotal_clicked);
	connect(ui->pbtn_exit, &QPushButton::clicked,
		this, &PictureViewerThumbnails::pbtn_exit_clicked);
	connect(ui->pbtn_nextPicture, &QPushButton::clicked,
		this, &PictureViewerThumbnails::pbtn_nextPicture_clicked);
	connect(ui->pbtn_prePicture, &QPushButton::clicked,
		this, &PictureViewerThumbnails::pbtn_prePicture_clicked);
}

void PictureViewerThumbnails::loadImageList()
{
	_listWidget->clear();
	m_imageFiles.clear();

	QDir dir(m_rootPath);
	QStringList nameFilters;
	const auto formats = QImageReader::supportedImageFormats();
	for (const QByteArray& fmt : formats)
		nameFilters << "*." + fmt;

	QFileInfoList fileList = dir.entryInfoList(nameFilters, QDir::Files | QDir::NoSymLinks | QDir::Readable, QDir::Name);
	for (const QFileInfo& fileInfo : fileList) {
		m_imageFiles << fileInfo.absoluteFilePath();
		QListWidgetItem* item = new QListWidgetItem();
		item->setText(fileInfo.fileName());
		item->setData(Qt::UserRole, fileInfo.absoluteFilePath());
		item->setSizeHint(QSize(m_thumbnailSize.width() + 16, m_thumbnailSize.height() + 32));
		_listWidget->addItem(item);
		loadThumbnail(fileInfo.absoluteFilePath(), item);
	}
}

void PictureViewerThumbnails::loadThumbnail(const QString& imagePath, QListWidgetItem* item)
{
	QImageReader reader(imagePath);
	reader.setAutoTransform(true);
	QImage img = reader.read();
	if (!img.isNull()) {
		QImage scaledImg = img.scaled(m_thumbnailSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

		QPixmap squarePixmap(m_thumbnailSize);
		squarePixmap.fill(Qt::transparent);

		QPainter painter(&squarePixmap);
		int x = (m_thumbnailSize.width() - scaledImg.width()) / 2;
		int y = (m_thumbnailSize.height() - scaledImg.height()) / 2;
		painter.drawImage(x, y, scaledImg);
		painter.end();

		item->setIcon(QIcon(squarePixmap));
	}
	else {
		item->setIcon(QIcon());
	}
}

void PictureViewerThumbnails::onThumbnailSelected()
{
	QListWidgetItem* item = _listWidget->currentItem();
	if (item) {
		QString path = item->data(Qt::UserRole).toString();
		ui->statusBar->showMessage(path);
	}
}

void PictureViewerThumbnails::pbtn_exit_clicked()
{
	this->close();
}

void PictureViewerThumbnails::pbtn_deleteTotal_clicked()
{
	auto result = QMessageBox::question(this, "提示", "是否删除当前目录下所有图片？", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
	if (result == QMessageBox::No) {
		return;
	}

	// 获取当前选中目录的绝对路径
	QString categoryPath = m_rootPath;
	QDir dir(categoryPath);

	// 检查目录是否存在
	if (!dir.exists()) {
		qDebug() << "Directory does not exist:" << categoryPath;
		return;
	}

	// 删除目录中的所有文件
	QStringList fileFilters;
	fileFilters << "*"; // 匹配所有文件
	QStringList files = dir.entryList(fileFilters, QDir::Files | QDir::NoSymLinks);
	for (const QString& file : files) {
		QString filePath = dir.filePath(file);
		if (!QFile::remove(filePath)) {
			qDebug() << "Failed to delete file:" << filePath;
		}
		else {
			qDebug() << "Deleted file:" << filePath;
		}
	}

	// 删除目录中的所有子文件夹及其内容
	QStringList subDirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& subDir : subDirs) {
		QDir subDirPath(dir.filePath(subDir));
		if (!subDirPath.removeRecursively()) {
			qDebug() << "Failed to delete subdirectory:" << subDirPath.path();
		}
		else {
			qDebug() << "Deleted subdirectory:" << subDirPath.path();
		}
	}

	qDebug() << "Cleared all contents of directory:" << categoryPath;

	// 更新目录列表
	loadImageList();
}

void PictureViewerThumbnails::pbtn_delete_clicked()
{
	// 获取当前选中的索引
	QModelIndex currentIndex = _listWidget->currentIndex();

	// 检查索引是否有效
	if (!currentIndex.isValid()) {
		qDebug() << "No picture selected for deletion.";
		return;
	}

	int row = currentIndex.row();

	// 获取当前选中图片的绝对路径
	QString picturePath = currentIndex.data(Qt::UserRole).toString();

	// 删除文件
	QFile file(picturePath);
	if (file.exists()) {
		if (file.remove()) {
			qDebug() << "Deleted picture:" << picturePath;
		}
		else {
			qDebug() << "Failed to delete picture:" << picturePath;
			return;
		}
	}
	else {
		qDebug() << "File does not exist:" << picturePath;
		return;
	}

	// 从m_imageFiles中移除
	m_imageFiles.removeAll(picturePath);

	// 从listWidget中移除item
	QListWidgetItem* item = _listWidget->takeItem(row);
	delete item;

	// 设置当前索引为删除项之前的索引，如果有的话
	int newRow = row - 1;
	if (newRow < 0 && _listWidget->count() > 0) {
		newRow = 0;
	}
	if (newRow >= 0 && _listWidget->count() > 0) {
		_listWidget->setCurrentRow(newRow);
	}
}

void PictureViewerThumbnails::pbtn_prePicture_clicked()
{
	int count = _listWidget->count();
	if (count == 0) return;

	int currentRow = _listWidget->currentRow();
	if (currentRow == -1) {
		// 没有选中任何项，默认选中第一个
		_listWidget->setCurrentRow(0);
		return;
	}
	if (currentRow > 0) {
		_listWidget->setCurrentRow(currentRow - 1);
	}
	// 如果已经是第一个，则不做处理（也可循环到最后一个，根据需求调整）
}

void PictureViewerThumbnails::pbtn_nextPicture_clicked()
{
	int count = _listWidget->count();
	if (count == 0) return;

	int currentRow = _listWidget->currentRow();
	if (currentRow == -1) {
		// 没有选中任何项，默认选中第一个
		_listWidget->setCurrentRow(0);
		return;
	}
	if (currentRow < count - 1) {
		_listWidget->setCurrentRow(currentRow + 1);
	}
	// 如果已经是最后一个，则不做处理（也可循环到第一个，根据需求调整）
}
