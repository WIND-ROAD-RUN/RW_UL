#pragma once

#include"gtest/gtest.h"

#include <QApplication>
#include <QLabel>

#include "ime_ModelEngineFactory.h"
#include "imgPro_ImageProcess.hpp"
#include "imgPro_ImageProcessUtilty.hpp"


class ImageProcessTest : public ::testing::Test {
protected:
	void SetUp() override {
		createImgPro();
		iniGetIndexContext();
		iniEliminationInfoGetConfig();
		iniEliminationContext();
		iniDefectResultGetConfig();
		iniDefectDrawConfig();
	}

	void createImgPro()
	{
		config.modelPath = R"(C:\Users\rw\Desktop\models\niukou.engine)";
		engine = rw::ModelEngineFactory::createModelEngine(
			config, rw::ModelType::Yolov11_Seg, rw::ModelEngineDeployType::TensorRT);
		imgProcess = std::make_unique<rw::imgPro::ImageProcess>(engine);
	}

	void iniEliminationContext()
	{
		auto& context = imgProcess->getContext();
		//context.indexGetContext.removeIndicesIf = [](rw::imgPro::ClassId classId, rw::imgPro::ProcessResultIndex index) {
		//	return classId == 1;
		//	};


		context.eliminationInfoGetContext.getEliminationItemFuncSpecialOperator = [this](rw::imgPro::EliminationItem& item,
			const rw::DetectionRectangleInfo& info,
			const rw::imgPro::EliminationInfoGetConfig& cfg) {
				item.customFields["location"] = 180;
				item.customFields["descrption"] = "asdwa";
				item.customFields["cout"] = 18.5f;
				auto c = item.customFields["cout"];
			if (c.has_value())
			{
				const auto& coutValue = std::any_cast<float>(c);
			}
			
			if (info.width>10000)
			{
				item.isBad = true;
			}

			};
	}

	void iniGetIndexContext(){
		auto& context = imgProcess->getContext();
		//context.indexGetContext.removeIndicesIf = [](rw::imgPro::ClassId classId, rw::imgPro::ProcessResultIndex index) {
		//	return classId == 1;
		//	};


		context.indexGetContext.removeIndicesIfByInfo = [this](const rw::DetectionRectangleInfo& info) {
			return false;
			};
	}

	void iniEliminationInfoGetConfig()
	{
		auto& context = imgProcess->getContext();
		rw::imgPro::EliminationInfoGetConfig eliminationInfoGetConfig;
		eliminationInfoGetConfig.areaFactor = 0.00157;
		eliminationInfoGetConfig.scoreFactor = 100;
		eliminationInfoGetConfig.isUsingArea = false;
		eliminationInfoGetConfig.isUsingScore = true;
		eliminationInfoGetConfig.scoreRange = { 0,100 };
		eliminationInfoGetConfig.scoreIsUsingComplementarySet = false;
		eliminationInfoGetConfigs[0] = eliminationInfoGetConfig;
		eliminationInfoGetConfigs[1] = eliminationInfoGetConfig;
		eliminationInfoGetConfigs[2] = eliminationInfoGetConfig;
		eliminationInfoGetConfigs[3] = eliminationInfoGetConfig;
		eliminationInfoGetConfigs[4] = eliminationInfoGetConfig;
		eliminationInfoGetConfigs[5] = eliminationInfoGetConfig;
		eliminationInfoGetConfigs[6] = eliminationInfoGetConfig;
		eliminationInfoGetConfigs[7] = eliminationInfoGetConfig;
		context.eliminationCfg = eliminationInfoGetConfigs;
	}
	void iniDefectResultGetConfig()
	{
		auto& context = imgProcess->getContext();
		rw::imgPro::DefectResultInfoFunc::DefectResultGetConfig defectConfig;
		defectConfig.isEnable = false;
		defectConfigs[0] = defectConfig;
		defectConfig.isEnable = true;
		defectConfigs[1] = defectConfig;
		defectConfigs[2] = defectConfig;
		defectConfigs[3] = defectConfig;
		defectConfigs[4] = defectConfig;
		defectConfigs[5] = defectConfig;
		defectConfigs[6] = defectConfig;
		defectConfigs[7] = defectConfig;
		defectConfigs[8] = defectConfig;
		defectConfigs[9] = defectConfig;
		context.defectCfg = defectConfigs;

	}
	void iniDefectDrawConfig()
	{
		auto& context = imgProcess->getContext();
		rw::imgPro::DefectDrawFunc::DefectDrawConfig drawConfig;
		drawConfig.isDrawDefects = true;
		drawConfig.isDrawDisableDefects = true;
		context.defectDrawCfg = drawConfig;

		context.runTextConfig.isDrawExtraText = true;
		context.runTextConfig.isDisOperatorTime = false;
		context.runTextConfig.isDisProcessImgTime = true;
	}
public:
	int left = 100;
	int right = 300;
	rw::ModelEngineConfig config;
	std::unique_ptr<rw::ModelEngine> engine;
	std::unique_ptr<rw::imgPro::ImageProcess> imgProcess;
	rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap eliminationInfoGetConfigs;
	rw::imgPro::DefectResultInfoFunc::ClassIdWithConfigMap defectConfigs;
};