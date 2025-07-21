#include"imgPro_ImageProcess.hpp"

#include "imgPro_IndexFunc.hpp"
#include "rqw_CameraObjectCore.hpp"

namespace rw
{
	namespace imgPro
	{
		ImageProcess::ImageProcess(std::unique_ptr<rw::ModelEngine>& engine)
		{
			_engine = std::move(engine);
		}

		ImageProcess::~ImageProcess()
		{
			_engine.reset();
		}

		std::unique_ptr<rw::ModelEngine>& ImageProcess::getModelEngine()
		{
			return _engine;
		}

		ProcessResult ImageProcess::processImg(const cv::Mat& mat)
		{
			if (!_engine) {
				throw std::runtime_error("Model engine is not initialized.");
			}
			auto processResult = _engine->processImg(mat);
			_processResult = processResult;
			return _processResult;
		}

		ProcessResultIndexMap ImageProcess::getIndex(const ProcessResult& processResult)
		{
			auto indexMap = rw::imgPro::IndexFunc::getIndex(processResult);
			_processResultIndexMap = indexMap;
			return _processResultIndexMap;
		}

		EliminationInfo ImageProcess::getEliminationInfo(const ProcessResult& processResult,
			const ProcessResultIndexMap& indexMap,
			const rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap& configs)
		{
			auto eliminationInfo = rw::imgPro::EliminationInfoFunc::getEliminationInfo(processResult, indexMap, configs);
			_eliminationInfo = eliminationInfo;
			return _eliminationInfo;
		}

		DefectResultInfo ImageProcess::getDefectResultInfo(const EliminationInfo& eliminationInfo,
			const rw::imgPro::DefectResultInfoFunc::ClassIdWithConfigMap& configs)
		{
			auto defectResultInfo = rw::imgPro::DefectResultInfoFunc::getDefectResultInfo(eliminationInfo, configs);
			_defectResultInfo = defectResultInfo;
			return _defectResultInfo;
		}

		void ImageProcess::operator()(const cv::Mat& mat)
		{
			processImg(mat);
			getIndex(_processResult);
			getEliminationInfo(_processResult, _processResultIndexMap, _context.eliminationCfg);
			getDefectResultInfo(_eliminationInfo, _context.defectCfg);
		}

		QImage ImageProcess::getMaskImg(const cv::Mat& mat) const
		{
			auto img = rw::rqw::cvMatToQImage(mat);
			rw::imgPro::DefectDrawFunc::drawDefectRecs(img, _defectResultInfo, _processResult, _context.defectDrawCfg);
			return img;
		}
	}
}
