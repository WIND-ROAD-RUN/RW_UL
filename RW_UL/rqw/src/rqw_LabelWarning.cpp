#include "rqw_LabelWarning.h"

namespace rw
{
	namespace rqw
	{
		void LabelWarning::build_ui()
		{
			// 初始化样式
			this->setStyleSheet(QString("QLabel { color: %1; }").arg(_warningColor));
			this->warningInfoList = new WarningInfoList(this);
		}

		void LabelWarning::build_connect()
		{
			// 连接定时器信号到槽函数
			connect(_timerToGray, &QTimer::timeout, this, &LabelWarning::onTimeoutToGray);
			connect(_timerToBlack, &QTimer::timeout, this, &LabelWarning::onTimeoutToBlack);
			connect(this, &ClickableLabel::clicked, this, &LabelWarning::labelClicked);
			connect(warningInfoList, &WarningInfoList::clearWarnings, this, &LabelWarning::clearWarningHistory);
		}

		LabelWarning::LabelWarning(QWidget* parent)
			: ClickableLabel(parent),
			_timerToGray(new QTimer(this)),
			_timerToBlack(new QTimer(this)),
			_maxHistorySize(100),
			_warningColor("red"),
			_timeoutColor("gray"),
			_grayDuration(60000) // 默认灰色持续时间为 60 秒
		{
			build_ui();
			build_connect();
		}

		void LabelWarning::addWarning(const QString& message, int redDuration)
		{
			// 获取当前时间戳
			QDateTime timestamp = QDateTime::currentDateTime();

			// 添加到历史队列
			_history.emplace_back(timestamp, message);

			// 检查队列容量
			if (_history.size() > _maxHistorySize) {
				_history.pop_front(); // 移除最早的警告信息
			}

			// 更新当前警告信息
			_currentMessage = message;
			this->setText(_currentMessage);

			// 设置文字为警告颜色
			this->setStyleSheet(QString("QLabel { color: %1; }").arg(_warningColor));

			// 启动红色到灰色的定时器
			_timerToGray->start(redDuration);

			// 停止灰色到黑色的定时器（如果正在运行）
			_timerToBlack->stop();
		}

		void LabelWarning::addWarning(const QString& message, bool updateTimestampIfSame, int redDuration)
		{
			// 如果启用了更新时间戳的功能，并且当前警告信息与上一次相同
			if (updateTimestampIfSame && !_history.empty() && _history.back().second == message) {
				// 更新最后一条警告信息的时间戳
				_history.back().first = QDateTime::currentDateTime();

				// 更新当前警告信息
				_currentMessage = message;
				this->setText(_currentMessage);

				// 重置红色到灰色的定时器
				_timerToGray->start(redDuration);

				// 停止灰色到黑色的定时器（如果正在运行）
				_timerToBlack->stop();

				return;
			}

			// 如果信息不同或未启用时间戳更新功能，按正常逻辑添加警告信息
			addWarning(message, redDuration);
		}

		void LabelWarning::setMaxHistorySize(size_t maxSize)
		{
			_maxHistorySize = maxSize;

			// 如果当前队列超过新设置的容量，移除多余的元素
			while (_history.size() > _maxHistorySize) {
				_history.pop_front();
			}
		}

		std::deque<std::pair<QDateTime, QString>> LabelWarning::getHistory() const
		{
			return _history;
		}

		void LabelWarning::setWarningColor(const QString& color)
		{
			_warningColor = color;
		}

		void LabelWarning::setTimeoutColor(const QString& color)
		{
			_timeoutColor = color;
		}

		void LabelWarning::setGrayDuration(int duration)
		{
			_grayDuration = duration;
		}

		void LabelWarning::onTimeoutToGray()
		{
			// 停止红色到灰色的定时器
			_timerToGray->stop();

			// 将文字颜色变为灰色
			this->setStyleSheet(QString("QLabel { color: %1; }").arg(_timeoutColor));

			// 启动灰色到黑色的定时器
			_timerToBlack->start(_grayDuration);
		}

		void LabelWarning::onTimeoutToBlack()
		{
			// 停止灰色到黑色的定时器
			_timerToBlack->stop();

			// 将文字变为黑色并显示 "暂无报警"
			this->setText("暂无报警");
			this->setStyleSheet("QLabel { color: black; }");
		}

		void LabelWarning::labelClicked()
		{
			warningInfoList->setWarningHistory(_history);
			warningInfoList->setWindowFlags(Qt::Window | Qt::CustomizeWindowHint);
			warningInfoList->show();
		}

		void LabelWarning::clearWarningHistory()
		{
			_history.clear();
		}
	}
}