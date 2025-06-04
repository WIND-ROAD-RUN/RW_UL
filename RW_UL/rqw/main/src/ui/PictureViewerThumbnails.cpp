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
	connect(_listWidget, &QListWidget::itemSelectionChanged, this, &PictureViewerThumbnails::onThumbnailSelected);

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
