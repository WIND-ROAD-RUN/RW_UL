#pragma once

#include <QLabel>
#include <QTimer>
#include <QDateTime>
#include <deque>
#include "rqw_LabelClickable.h"
#include "WarningInfoList.h"

namespace rw
{
	namespace rqw
	{
		class LabelWarning : public ClickableLabel
		{
			Q_OBJECT
		private:
			WarningInfoList* warningInfoList;

		private:
			void build_ui();
			void build_connect();

		public:
			explicit LabelWarning(QWidget* parent = nullptr);

			// 添加警告信息
			void addWarning(const QString& message, int redDuration = 5000);
			void addWarning(const QString& message, bool updateTimestampIfSame, int redDuration = 5000);

			// 设置队列最大容量
			void setMaxHistorySize(size_t maxSize);

			// 获取历史警告信息
			std::deque<std::pair<QDateTime, QString>> getHistory() const;

			// 设置警告颜色
			void setWarningColor(const QString& color);

			// 设置超时后的颜色
			void setTimeoutColor(const QString& color);

			// 设置灰色状态持续时间
			void setGrayDuration(int duration);

		private slots:
			// 槽函数：将文字颜色变为灰色
			void onTimeoutToGray();

			// 槽函数：将文字变为黑色 "暂无报警"
			void onTimeoutToBlack();

			void labelClicked();

		private slots:
			void clearWarningHistory(); // 清空历史记录

		private:
			QTimer* _timerToGray;    // 定时器，用于控制红色变灰色
			QTimer* _timerToBlack;   // 定时器，用于控制灰色变黑色
			QString _currentMessage; // 当前警告信息
			std::deque<std::pair<QDateTime, QString>> _history; // 历史警告信息队列
			size_t _maxHistorySize;  // 队列最大容量

			QString _warningColor;   // 警告颜色
			QString _timeoutColor;   // 超时后的颜色
			int _grayDuration;       // 灰色状态持续时间（毫秒）
		};
	}
}