#pragma once

#include <QObject>

#include"cdm_AiModelConfigIndex.h"
#include"imeot_ModelEngineOT.h"
#include"imest_ModelEnginest.h"

class ModelStorageManager : public QObject
{
	Q_OBJECT
public:
	QString imageSavePath;
private:
	rw::cdm::AiModelConfigIndex _config_index{};
public:
	rw::cdm::AiModelConfigIndex getModelConfigIndex();
public:
	void addModelConfig(const rw::cdm::AiModelConfig& item);

	void eraseModelConfig(const rw::cdm::AiModelConfig& item);

	void saveIndexConfig();
public:
	ModelStorageManager(QObject* parent);
	~ModelStorageManager();
public:
	void setRootPath(QString path);
	QString getRootPath();
private:
	void build_manager();
	void destroy_manager();
private:
	void build_tempDirectory();
public:
	void clear_temp();
public:
	void check_work_temp(const QString& imageRootPath);
private:
	void check_work1Temp(const QString& imageRootPath);
public:
	void save_work1_image(const QImage& image, bool isgood);
private:
	void saveImageWithTimestamp(const QImage& image, const QString& folder);
public:
	QVector<QString> getBadImagePathList();
	QVector<QString> getGoodImagePathList();
public:
	size_t work1_good_count_;
	size_t work1_bad_count_;
private:
	void check_work2Temp(const QString& imageRootPath);
public:
	void save_work2_image(const QImage& image, bool isgood);
public:
	size_t work2_good_count_;
	size_t work2_bad_count_;
private:
	void check_work3Temp(const QString& imageRootPath);
public:
	void save_work3_image(const QImage& image, bool isgood);
public:
	size_t work3_good_count_;
	size_t work3_bad_count_;
private:
	void check_work4Temp(const QString& imageRootPath);
public:
	void save_work4_image(const QImage& image, bool isgood);
public:
	size_t work4_good_count_;
	size_t work4_bad_count_;

private:
	QString _rootPath;
};
