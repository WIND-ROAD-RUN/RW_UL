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
	updateCategoryList();
	QDir rootDir(m_rootPath);
	preloadAllCategoryImages(rootDir);
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

	_loadingDialog = new LoadingDialog(this);
	_categoryModel = new QStandardItemModel(this);
	pictureViewerUtilty = new PictureViewerUtilty(this);
	ui->treeView_categoryTree->setModel(_categoryModel);
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

	connect(ui->treeView_categoryTree->selectionModel(), &QItemSelectionModel::currentChanged,
		this, [this](const QModelIndex& current, const QModelIndex&) {
			Q_UNUSED(current);
			loadImageList();
		});

	connect(_listWidget, &QListWidget::itemDoubleClicked,
		this, &PictureViewerThumbnails::onThumbnailDoubleClicked);
}

void PictureViewerThumbnails::loadImageList()
{
	_listWidget->clear();
	m_imageFiles.clear();

	// 获取当前选中目录的路径
	QModelIndex currentCategoryIndex = ui->treeView_categoryTree->currentIndex();
	QString dirPath = m_rootPath;
	if (currentCategoryIndex.isValid()) {
		QVariant data = currentCategoryIndex.data(Qt::UserRole);
		if (data.isValid()) {
			dirPath = data.toString();
		}
	}

	// 从缓存获取图片路径
	QStringList imageList = m_categoryImageCache.value(dirPath);

	for (const QString& imagePath : imageList) {
		m_imageFiles << imagePath;
		QFileInfo fileInfo(imagePath);
		QListWidgetItem* item = new QListWidgetItem();
		item->setText(fileInfo.fileName());
		item->setData(Qt::UserRole, imagePath);
		item->setSizeHint(QSize(m_thumbnailSize.width() + 16, m_thumbnailSize.height() + 32));
		_listWidget->addItem(item);
		loadThumbnail(imagePath, item);
	}
}

void PictureViewerThumbnails::loadThumbnail(const QString& imagePath, QListWidgetItem* item)
{
	if (m_thumbnailCache.contains(imagePath)) {
		item->setIcon(QIcon(m_thumbnailCache.value(imagePath)));
		return;
	}
	// 兜底：如果没有缓存则正常加载
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
		m_thumbnailCache.insert(imagePath, squarePixmap);
	}
	else {
		item->setIcon(QIcon());
	}
}

void PictureViewerThumbnails::updateCategoryList()
{
	_categoryModel->clear();
	// 检查 _rootPath 是否有效
	if (m_rootPath.isEmpty()) {
		qDebug() << "Root path is empty.";
		return;
	}

	QDir rootDir(m_rootPath);
	if (!rootDir.exists()) {
		qDebug() << "Root path does not exist:" << m_rootPath;
		return;
	}

	_categoryModel->setHorizontalHeaderLabels({ "文件树" });

	// 遍历 _rootPath 下的所有文件夹
	QStringList categoryFolders = rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& category : categoryFolders) {
		QString categoryPath = rootDir.filePath(category);
		QStandardItem* categoryItem = new QStandardItem(category);

		// 保存绝对路径到节点
		categoryItem->setData(categoryPath, Qt::UserRole);

		// 递归添加子文件夹
		QDir categoryDir(categoryPath);
		addSubFolders(categoryDir, categoryItem);

		_categoryModel->appendRow(categoryItem);
	}

	// 设置模型到 treeView_categoryTree

	ui->treeView_categoryTree->expandAll();

	// 设置当前选中的索引为第一个最深的子节点
	QModelIndex firstDeepestIndex = findFirstDeepestIndex(_categoryModel);
	if (firstDeepestIndex.isValid()) {
		ui->treeView_categoryTree->setCurrentIndex(firstDeepestIndex);
	}
}

void PictureViewerThumbnails::addSubFolders(const QDir& parentDir, QStandardItem* parentItem)
{
	// 获取当前目录下的所有子文件夹
	QStringList subFolders = parentDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& subFolder : subFolders) {
		QString subFolderPath = parentDir.filePath(subFolder);
		QStandardItem* subFolderItem = new QStandardItem(subFolder);

		// 保存绝对路径到节点
		subFolderItem->setData(subFolderPath, Qt::UserRole);

		// 递归处理子文件夹
		QDir subFolderDir(subFolderPath);
		addSubFolders(subFolderDir, subFolderItem);

		parentItem->appendRow(subFolderItem);
	}
}

QModelIndex PictureViewerThumbnails::findFirstDeepestIndex(QStandardItemModel* model)
{
	if (!model || model->rowCount() == 0) {
		return QModelIndex();
	}

	QStandardItem* rootItem = model->invisibleRootItem();
	return findDeepestChild(rootItem);
}

QModelIndex PictureViewerThumbnails::findDeepestChild(QStandardItem* parentItem)
{
	if (!parentItem || parentItem->rowCount() == 0) {
		return parentItem ? parentItem->index() : QModelIndex();
	}

	// 遍历子节点，找到第一个最深的子节点
	QStandardItem* firstChild = parentItem->child(0);
	return findDeepestChild(firstChild);
}

void PictureViewerThumbnails::preloadAllCategoryImages(const QDir& dir)
{
	QStringList nameFilters;
	const auto formats = QImageReader::supportedImageFormats();
	for (const QByteArray& fmt : formats)
		nameFilters << "*." + fmt;

	QFileInfoList fileList = dir.entryInfoList(nameFilters, QDir::Files | QDir::NoSymLinks | QDir::Readable, QDir::Name);
	QStringList imagePaths;
	for (const QFileInfo& fileInfo : fileList) {
		QString imagePath = fileInfo.absoluteFilePath();
		imagePaths << imagePath;

		// 只在缓存中没有时才生成缩略图
		if (!m_thumbnailCache.contains(imagePath)) {
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

				m_thumbnailCache.insert(imagePath, squarePixmap);
			}
		}
	}
	m_categoryImageCache.insert(dir.absolutePath(), imagePaths);

	// 递归处理子目录
	QStringList subDirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& subDir : subDirs) {
		QDir subDirPath(dir.filePath(subDir));
		preloadAllCategoryImages(subDirPath);
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
	emit viewerClosed();
}

void PictureViewerThumbnails::pbtn_deleteTotal_clicked()
{
	auto result = QMessageBox::question(this, "提示", "是否删除当前目录下所有图片？", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
	if (result == QMessageBox::No) {
		return;
	}

	// 获取当前选中目录的绝对路径
	QModelIndex currentCategoryIndex = ui->treeView_categoryTree->currentIndex();
	QString categoryPath = m_rootPath;
	if (currentCategoryIndex.isValid()) {
		QVariant data = currentCategoryIndex.data(Qt::UserRole);
		if (data.isValid()) {
			categoryPath = data.toString();
		}
	}
	QDir dir(categoryPath);

	// 检查目录是否存在
	if (!dir.exists()) {
		qDebug() << "Directory does not exist:" << categoryPath;
		return;
	}

	// 删除目录中的所有文件，并同步移除缩略图缓存
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
			m_thumbnailCache.remove(filePath); // 移除缩略图缓存
		}
	}

	// 删除目录中的所有子文件夹及其内容
	QStringList subDirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	for (const QString& subDir : subDirs) {
		QDir subDirPath(dir.filePath(subDir));
		// 递归删除子目录下的图片缩略图缓存
		QStringList subFiles = subDirPath.entryList(fileFilters, QDir::Files | QDir::NoSymLinks);
		for (const QString& subFile : subFiles) {
			QString subFilePath = subDirPath.filePath(subFile);
			m_thumbnailCache.remove(subFilePath);
		}
		if (!subDirPath.removeRecursively()) {
			qDebug() << "Failed to delete subdirectory:" << subDirPath.path();
		}
		else {
			qDebug() << "Deleted subdirectory:" << subDirPath.path();
		}
	}

	// 清除该目录及其子目录的图片路径缓存
	auto it = m_categoryImageCache.begin();
	while (it != m_categoryImageCache.end()) {
		if (it.key().startsWith(categoryPath)) {
			it = m_categoryImageCache.erase(it);
		}
		else {
			++it;
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

	// 从缩略图缓存中移除
	m_thumbnailCache.remove(picturePath);

	// 从目录图片路径缓存中移除
	QModelIndex currentCategoryIndex = ui->treeView_categoryTree->currentIndex();
	QString dirPath = m_rootPath;
	if (currentCategoryIndex.isValid()) {
		QVariant data = currentCategoryIndex.data(Qt::UserRole);
		if (data.isValid()) {
			dirPath = data.toString();
		}
	}
	if (m_categoryImageCache.contains(dirPath)) {
		m_categoryImageCache[dirPath].removeAll(picturePath);
	}

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

void PictureViewerThumbnails::onThumbnailDoubleClicked(QListWidgetItem* item)
{
	if (!item) return;
	QString imagePath = item->data(Qt::UserRole).toString();
	pictureViewerUtilty->setImgPath(imagePath);
	pictureViewerUtilty->show();
}
