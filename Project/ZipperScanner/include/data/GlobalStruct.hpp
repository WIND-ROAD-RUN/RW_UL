#pragma once

#include<memory>
#include<QString>
#include<QObject>

class GlobalStructData
	:public QObject
{
	Q_OBJECT
public:
	static GlobalStructData& getInstance()
	{
		static GlobalStructData instance;
		return instance;
	}

	GlobalStructData(const GlobalStructData&) = delete;
	GlobalStructData& operator=(const GlobalStructData&) = delete;
private:
	GlobalStructData();
	~GlobalStructData() = default;
};