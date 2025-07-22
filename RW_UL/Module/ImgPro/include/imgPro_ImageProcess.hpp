#pragma once

#include"ime_ModelEngine.h"
#include "imgPro_DefectDrawFunc.hpp"
#include "imgPro_DefectResultInfoFunc.hpp"
#include "imgPro_EliminationInfoFunc.hpp"
#include "imgPro_IndexFunc.hpp"

namespace rw
{
	namespace imgPro
	{
		using ClassIds = std::vector<ClassId>;

		struct ImageProcessContext
		{
		public:
			EliminationInfoFunc::ClassIdWithConfigMap eliminationCfg{};
			DefectResultInfoFunc::ClassIdWithConfigMap defectCfg{};
			DefectDrawFunc::DefectDrawConfig defectDrawCfg{};
			DefectDrawFunc::RunTextConfig runTextConfig{};
			IndexGetContext indexGetContext{};
			EliminationInfoGetContext eliminationInfoGetContext{};
		};

		class ImageProcess
		{
		public:
			explicit ImageProcess(std::unique_ptr<rw::ModelEngine>& engine);
			~ImageProcess();
		private:
			ImageProcessContext _context{};
		public:
			ImageProcessContext& getContext()
			{
				return _context;
			}
		private:
			std::unique_ptr<rw::ModelEngine> _engine = nullptr;
		public:
			std::unique_ptr<rw::ModelEngine>& getModelEngine();
		private:
			RunTime _processImgTime{};
			RunTime _operatorTime{};
		public:
			ProcessResult processImg(const cv::Mat & mat);
			ProcessResultIndexMap getIndex(const ProcessResult & processResult);
			EliminationInfo getEliminationInfo(const ProcessResult& processResult, const ProcessResultIndexMap& indexMap, const rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap& configs);
			DefectResultInfo getDefectResultInfo(const EliminationInfo& eliminationInfo, const rw::imgPro::DefectResultInfoFunc::ClassIdWithConfigMap& configs);
		public:
			void operator()(const cv::Mat& mat);
			QImage getMaskImg(const cv::Mat& mat);
		private:
			ProcessResultIndexMap _processResultIndexMap{};
			ProcessResult _processResult{};
			EliminationInfo _eliminationInfo{};
			DefectResultInfo _defectResultInfo{};
		public:
			const ProcessResultIndexMap& getProcessResultIndexMap() const
			{
				return _processResultIndexMap;
			}
			const ProcessResult& getProcessResult() const
			{
				return _processResult;
			}
			const EliminationInfo& getEliminationInfo() const
			{
				return _eliminationInfo;
			}
			const DefectResultInfo& getDefectResultInfo() const
			{
				return _defectResultInfo;
			}
		};
	}
}
