#pragma once

#include <QDialog>
#include <QLabel>
#include <QStandardItemModel>

QT_BEGIN_NAMESPACE
namespace Ui { class PicturesPainterClass; };
QT_END_NAMESPACE

struct RectangeConfig
{
	int classid;		//框的id
	QColor color;		//绘画框的颜色
	QString name;		//会话框的名字
	QString descrption;	//会话框的描述
};

// 绘画类
class DrawLabel : public QLabel
{
	Q_OBJECT
public:
	explicit DrawLabel(QWidget* parent = nullptr);
	QRectF getNormalizedRect() const;
signals:
	void rectSelected(const QRectF& normalizedRect);
protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void paintEvent(QPaintEvent* event) override;
public slots:
	void setDrawingEnabled(bool enabled);
private:
	QPoint m_startPoint;
	QPoint m_endPoint;
	bool m_isDrawing = false;
private:
	bool m_drawingEnabled = false; // 是否允许绘画
};

class PicturesPainter : public QDialog
{
	Q_OBJECT

public:
	PicturesPainter(QWidget* parent = nullptr);
	~PicturesPainter();

	// 绘画信息结构体
public:
	struct PainterRectangleInfo
	{
	public:
		using Point = std::pair<double, double>;
	public:
		Point leftTop{};
		Point rightTop{};
		Point leftBottom{};
		Point rightBottom{};
	public:
		double center_x{ -1 };
		double center_y{ -1 };
	public:
		double width{ -1 };
		double height{ -1 };
	public:
		long area{ -1 };
	public:
		size_t classId{ 0 };
		double score{ -1 };
	};

public:
	void setRectangleConfigs(const std::vector<RectangeConfig>& configs);
	void setDrawnRectangles(const std::vector<PainterRectangleInfo>& Rectangles);
	void setImage(const QImage& qImage);
	void setAspectRatio(double width, double height);

	QColor getColorByClassId(size_t size);
	QString getNameByClassId(size_t size);
	std::vector<PainterRectangleInfo> getRectangleConfigs();

private:
	void build_ui();
	void release_ui();
	void build_connect();

private:
	void showWhiteImageOnLabel(double width, double height);

protected:
	void showEvent(QShowEvent* event) override;

private slots:
	void btn_set_clicked();
	void btn_draw_clicked();
	void pbtn_ok_clicked();
	void pbtn_exit_clicked();
	void pbtn_setAspectRatio_clicked();
	void pbtn_openPicture_clicked();
	void btn_clear_clicked();

	void onListViewCurrentChanged(const QModelIndex& current, const QModelIndex& previous = QModelIndex());

	void onRectSelected(const QRectF& rect);
	void updateDrawLabel();

private:
	Ui::PicturesPainterClass* ui;
	double img_Width{ 0 };
	double img_Height{ 0 };

	std::vector<RectangeConfig> _configs{};
	std::vector<PainterRectangleInfo> _drawnRectangles{};

	QStandardItemModel* m_listModel = nullptr;   // 对应ListView
	QStandardItemModel* m_tableModel = nullptr;  // 对应TableView

	DrawLabel* drawLabel = nullptr;

	QRectF m_lastNormalizedRect; // 记录最后一次框选的标准化矩形
	bool m_hasSelectedRect = false; // 判断是否有有效区域

	bool isSetDrawnRectangles{ false };
	bool isSetAspectRatio{ false };
	bool isSetQImage{ false };

	QImage _qImage{ nullptr };
};
