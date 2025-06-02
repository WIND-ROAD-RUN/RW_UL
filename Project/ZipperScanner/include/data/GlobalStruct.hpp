#pragma once

#include<memory>
#include<QString>
#include<QObject>

#include "GeneralConfig.hpp"
#include "ScoreConfig.hpp"
#include "SetConfig.hpp"
#include "oso_StorageContext.hpp"

class GlobalStructDataZipper
	:public QObject
{
	Q_OBJECT
public:
	static GlobalStructDataZipper& getInstance()
	{
		static GlobalStructDataZipper instance;
		return instance;
	}

	GlobalStructDataZipper(const GlobalStructDataZipper&) = delete;
	GlobalStructDataZipper& operator=(const GlobalStructDataZipper&) = delete;
private:
	GlobalStructDataZipper();
	~GlobalStructDataZipper() = default;

public:
	void buildConfigManager(rw::oso::StorageType type);

public:
	cdm::GeneralConfig generalConfig;
	cdm::ScoreConfig scoreConfig;
	cdm::SetConfig setConfig;

	std::unique_ptr<rw::oso::StorageContext> storeContext{ nullptr };


	
};