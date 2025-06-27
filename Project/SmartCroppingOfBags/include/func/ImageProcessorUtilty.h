#pragma once
#include <QString>

#include "ime_utilty.hpp"
#include "rqw_HistoricalElementManager.hpp"
#include <Utilty.hpp>

using TimeFrameMatInfo = std::pair<Time, std::optional<rw::rqw::ElementInfo<cv::Mat>>>;
using TimeFrameQImageInfo = std::pair<Time, QImage>;


inline std::optional<std::chrono::system_clock::time_point> findTimeInterval(
	const std::vector<std::chrono::system_clock::time_point>& timeCollection,
	const std::chrono::system_clock::time_point& timePoint);


// 智能裁切吨袋检测信息
struct SmartCroppingOfBagsDefectInfo
{
public:
	// 缺陷
	struct DetectItem
	{
	public:
		double score = 0;	// 分数
		double area = 0;	// 面积
		int index = -1;		// 在processResult中的索引位置
		bool isBad = false;	// 是否满足剔废条件绘画红框
	};

	std::vector<DetectItem> heibaList;         // 黑疤
	std::vector<DetectItem> shudangList;       // 疏档
	std::vector<DetectItem> huapoList;         // 划破
	std::vector<DetectItem> jietouList;        // 接头
	std::vector<DetectItem> guasiList;         // 挂丝
	std::vector<DetectItem> podongList;        // 破洞
	std::vector<DetectItem> zangwuList;        // 脏污
	std::vector<DetectItem> noshudangList;     // 无疏档
	std::vector<DetectItem> modianList;        // 墨点
	std::vector<DetectItem> loumoList;         // 漏膜
	std::vector<DetectItem> xishudangList;     // 稀疏档
	std::vector<DetectItem> erweimaList;       // 二维码
	std::vector<DetectItem> damodianList;      // 大墨点
	std::vector<DetectItem> kongdongList;      // 孔洞
	std::vector<DetectItem> sebiaoList;        // 色标
	std::vector<DetectItem> yinshuaquexianList;// 印刷缺陷
	std::vector<DetectItem> xiaopodongList;    // 小破洞
	std::vector<DetectItem> jiaodaiList;       // 胶带
};

// 图片信息
struct MatInfo {
	rw::rqw::ElementInfo<cv::Mat> image;
	size_t index{};	// 拍照的相机的下标
	Time time;
	double location{};
public:
	// 默认构造函数
	MatInfo() = default;

	// 参数化构造函数
	MatInfo(const rw::rqw::ElementInfo<cv::Mat>& element) : image(element) {}

	// 拷贝构造函数
	MatInfo(const MatInfo& other)
		: image(other.image), index(other.index), time(other.time) {
	}

	// 拷贝赋值运算符（可选）
	MatInfo& operator=(const MatInfo& other) {
		if (this != &other) {
			image = other.image;
			index = other.index;
			time = other.time;
			location = other.location;
		}
		return *this;
	}
};

struct HistoryDetectInfo
{
public:
	bool hasCut{ false };
	size_t cutLocate{ 0 };
public:
	double bottomErrorLocation{ 0 };
	double topErrorLocation{0};
public:
	std::vector<rw::DetectionRectangleInfo> processResult;
public:
	QString processTime{};
public:
	HistoryDetectInfo() = default;
	HistoryDetectInfo(const std::vector<rw::DetectionRectangleInfo>& result) : processResult(result) {}
	// 拷贝构造函数
	HistoryDetectInfo(const HistoryDetectInfo& other)
		: processResult(other.processResult),
		hasCut(other.hasCut),
		cutLocate(other.cutLocate),
		processTime(other.processTime),
		bottomErrorLocation(other.bottomErrorLocation),
		topErrorLocation(other.topErrorLocation){
	}
	// 拷贝赋值运算符
	HistoryDetectInfo& operator=(const HistoryDetectInfo& other) {
		if (this != &other) {
			processResult = other.processResult;
			hasCut = other.hasCut;
			cutLocate = other.cutLocate;
			processTime = other.processTime;
			bottomErrorLocation = other.bottomErrorLocation;
			topErrorLocation = other.topErrorLocation;
		}
		return *this;
	}
};