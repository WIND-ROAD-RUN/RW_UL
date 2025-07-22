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
		iniDefectResultContext();
		iniDefectDrawConfig();
		iniRunTextConfig();
	}

	void iniRunTextConfig()
	{
		auto& context = imgProcess->getContext();

		context.runTextConfig.isDrawExtraText = true;
		context.runTextConfig.isDisOperatorTime = false;
		context.runTextConfig.isDisProcessImgTime = true;
		context.runTextConfig.extraTextColor = rw::rqw::RQWColor::Orange;
		context.runTextConfig.operatorTimeTextColor = rw::rqw::RQWColor::Gray;
		context.runTextConfig.processImgTimeTextColor = rw::rqw::RQWColor::Cyan;
		context.runTextConfig.runTextProportion = 0.08;
	}

	void createImgPro()
	{
		config.modelPath = R"(C:\Users\rw\Desktop\models\niukou.engine)";
		engine = rw::ModelEngineFactory::createModelEngine(
			config, rw::ModelType::Yolov11_Seg, rw::ModelEngineDeployType::TensorRT);
		imgProcess = std::make_unique<rw::imgPro::ImageProcess>(engine);
	}

	void iniDefectResultContext()
	{
		auto& context = imgProcess->getContext();
		context.defectResultGetContext.getDefectResultExtraOperate = [this](const rw::imgPro::EliminationItem& item) {
			auto find = item.customFields.find("someValueWillBeUsed");
			if (find!=item.customFields.end())
			{
				std::cout << "someValueWillBeUsed :"<<std::any_cast<int>(find->second)<<"score is:"<<item.score<<std::endl;
			}
			};
		context.defectResultGetContext.getDefectResultExtraOperateDisable = [this](const rw::imgPro::EliminationItem& item) {
			auto find = item.customFields.find("someValueWillBeUsed");
			if (find != item.customFields.end())
			{

				std::cout << "It 's no ,someValueWillBeUsed :" << std::any_cast<int>(find->second)-100 << "score is:" << item.score << std::endl;
			}
			};
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
				auto find = cfg.customFields.find("someValueWillBeUsed");
			if (find!= cfg.customFields.end())
			{
				auto value = std::any_cast<int>(find->second) + 100;
				item.customFields["someValueWillBeUsed"] = value;
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
		eliminationInfoGetConfig.scoreRange = { 0,80 };
		eliminationInfoGetConfig.scoreIsUsingComplementarySet = false;
		eliminationInfoGetConfig.customFields["someValueWillBeUsed"] = (int)(100);
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
		drawConfig.setAllIdsWithSameColor({ 0,1,2,3,4,5,6 }, rw::rqw::RQWColor::Brown, true);
		drawConfig.setAllIdsWithSameColor({ 0,1,2,3,4,5,6 }, rw::rqw::RQWColor::Blue, false);
		drawConfig.classIdWithColorWhichIsGood[2] = rw::rqw::RQWColor::Green;
		drawConfig.thickness = 20;
		drawConfig.fontSize = 20;
		context.defectDrawCfg = drawConfig;
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