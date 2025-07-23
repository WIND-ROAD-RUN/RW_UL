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
			IndexGetContext indexGetContext{};
		public:
			EliminationInfoFunc::ClassIdWithConfigMap eliminationCfg{};
			EliminationInfoGetContext eliminationInfoGetContext{};
		public:
			DefectResultInfoFunc::ClassIdWithConfigMap defectCfg{};
			DefectResultGetContext defectResultGetContext{};
		public:
			DefectDrawFunc::DefectDrawConfig defectDrawCfg{};
			DefectDrawFunc::RunTextConfig runTextCfg{};
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
			RunTime getProcessImgTime() const
			{
				return _processImgTime;
			}
			RunTime getOperatorTime() const
			{
				return _operatorTime;
			}
		public:
			ProcessResult processImg(const cv::Mat & mat);
			ProcessResultIndexMap getIndex(const ProcessResult & processResult);
			EliminationInfo getEliminationInfo(const ProcessResult& processResult, const ProcessResultIndexMap& indexMap, const rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap& configs);
			DefectResultInfo getDefectResultInfo(const EliminationInfo& eliminationInfo, const rw::imgPro::DefectResultInfoFunc::ClassIdWithConfigMap& configs);
		public:
			void operator()(const cv::Mat& mat);
			QImage getMaskImg(const cv::Mat& mat);
			static QImage getMaskImg(const cv::Mat& mat,
				const DefectResultInfo& defectResultInfo,
				const ProcessResult& processResult,
				ImageProcessContext& context,
				RunTime operatorTime,
				RunTime processImgTime
			);
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
