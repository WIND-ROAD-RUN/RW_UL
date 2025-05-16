#include"AutomaticAnnotationThread.h"

#include"ime_ModelEngineFactory.h"
#include"rqw_ImagePainter.h"
#include<QMessageBox>
#include<QFile>
#include<QFileInfo>
#include<QDir>
#include <QSet>
static std::vector<rw::DetectionRectangleInfo> filterByLabelList(
const std::vector<rw::DetectionRectangleInfo>& input,
const QVector<int>& labelList)
{
std::vector<rw::DetectionRectangleInfo> result;
QSet<int> labelSet(labelList.begin(), labelList.end()); // Corrected initialization of QSet

for (const auto& info : input) {
	if (labelSet.contains(static_cast<int>(info.classId))) {
		result.push_back(info);
	}
}
return result;
}

AutomaticAnnotationThread::AutomaticAnnotationThread(const QVector<QString>& imagePaths, QObject* parent)
    : QThread(parent), m_imagePaths(imagePaths) {

}

QString AutomaticAnnotationThread::getObjectDetectionDataSetItem(const std::vector<rw::DetectionRectangleInfo>& annotationDataSet, int width, int height)
{
	QString result;
	for (const auto& item : annotationDataSet)
	{
		std::string id = std::to_string(item.classId);

		double norCenterX = static_cast<double>(item.center_x) / static_cast<double>(width);
		double norCenterY = static_cast<double>(item.center_y) / static_cast<double>(height);
		double norWidth = static_cast<double>(item.width) / static_cast<double>(width);
		double norHeight = static_cast<double>(item.height) / static_cast<double>(height);

		auto textStr = id + " " +
			std::to_string(norCenterX) + " " + std::to_string(norCenterY) + " "
			+ std::to_string(norWidth) + " " + std::to_string(norHeight) + "\n"; // 添加换行符

		result.append(QString::fromStdString(textStr));
	}
	return result;
}

QString AutomaticAnnotationThread::getObjectSegmentDataSetItem(
	const std::vector<rw::DetectionRectangleInfo>& annotationDataSet, int width, int height)
{
	QString result;

	for (const auto& item : annotationDataSet)
	{
		std::string id = std::to_string(item.classId);

		// 归一化中心点和宽高
		double norCenterX = static_cast<double>(item.center_x) / static_cast<double>(width);
		double norCenterY = static_cast<double>(item.center_y) / static_cast<double>(height);
		double norWidth = static_cast<double>(item.width) / static_cast<double>(width);
		double norHeight = static_cast<double>(item.height) / static_cast<double>(height);

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
		auto textStr = id + pointsStr+"\n";

		// 添加到结果集
		result.append(QString::fromStdString(textStr));
	}

	return result;
}

void AutomaticAnnotationThread::saveLabels(const QString& label, const QString& fileName)
{
	QString labelPath = labelOutput + "/" + fileName + ".txt";
	QDir dir;
	if (!dir.exists(labelOutput)) {
		if (!dir.mkpath(labelOutput)) {
			qDebug() << "Failed to create directory:" << labelOutput;
			return;
		}
	}

	QFile file(labelPath);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		qDebug() << "Failed to open file for writing:" << labelPath;
		return;
	}

	QTextStream out(&file);
	out << label;
	file.close();
}

void AutomaticAnnotationThread::saveLabels_seg(const QString& label, const QString& fileName)
{
	
}

void AutomaticAnnotationThread::saveImage(const QString& imagePath)
{
	// 确保目标目录存在
	QDir dir;
	if (!dir.exists(imageOutput)) {
		if (!dir.mkpath(imageOutput)) {
			qDebug() << "Failed to create directory:" << imageOutput;
			return;
		}
	}

	// 获取源文件名
	QString fileName = QFileInfo(imagePath).fileName();

	// 构造目标路径
	QString targetPath = imageOutput + "/" + fileName;

	// 拷贝文件
	if (!QFile::copy(imagePath, targetPath)) {
		qDebug() << "Failed to copy file from" << imagePath << "to" << targetPath;
	}
	else {
		qDebug() << "File copied successfully to" << targetPath;
	}
}


void AutomaticAnnotationThread::run()
{
    auto engine = rw::ModelEngineFactory::createModelEngine(config, modelType, deployType);
    if (engine == nullptr)
    {
        return;
    }

    for (const QString& path : m_imagePaths) {
		auto mat = cv::imread(path.toStdString());
		if (mat.empty()) {
			qDebug() << "Failed to load image:" << path;
			continue;
		}
        auto result = engine->processImg(mat);
		result = filterByLabelList(result, labelList);

		auto fileName = QFileInfo(path).baseName();

		if (labelType==R"(Segment)")
		{
			auto label = getObjectSegmentDataSetItem(result, mat.cols, mat.rows);
			saveLabels(label, fileName);
		}
		else
		{
			auto label = getObjectDetectionDataSetItem(result, mat.cols, mat.rows);
			saveLabels(label, fileName);
		}
		saveImage(path);

        auto image=rw::rqw::ImagePainter::cvMatToQImage(mat);

		rw::rqw::ImagePainter::drawShapesOnSourceImg(image, result);

		QPixmap pixmap = QPixmap::fromImage(image);

        emit imageProcessed(path, pixmap);
    }

}
