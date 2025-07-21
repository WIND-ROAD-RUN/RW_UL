#include"imgPro_ImageProcess.hpp"

#include "imgPro_IndexFunc.hpp"
#include "rqw_CameraObjectCore.hpp"

#include <chrono> 

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

			auto start = std::chrono::high_resolution_clock::now();

			auto processResult = _engine->processImg(mat);

			auto end = std::chrono::high_resolution_clock::now();

			_processImgTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

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

			auto start = std::chrono::high_resolution_clock::now();

			processImg(mat);
			getIndex(_processResult);
			getEliminationInfo(_processResult, _processResultIndexMap, _context.eliminationCfg);
			getDefectResultInfo(_eliminationInfo, _context.defectCfg);

			auto end = std::chrono::high_resolution_clock::now();

			_operatorTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		}

		QImage ImageProcess::getMaskImg(const cv::Mat& mat)
		{
			auto img = rw::rqw::cvMatToQImage(mat);
			rw::imgPro::DefectDrawFunc::drawDefectRecs(img, _defectResultInfo, _processResult, _context.defectDrawCfg);

			_context.runTextConfig.operatorTimeText = QString::number(_operatorTime)+" ms";
			_context.runTextConfig.processImgTimeText = QString::number(_processImgTime) + " ms";

			auto & errorRecs = _defectResultInfo.defects;

			QVector<QString> errors;
			for ( auto & pairs: errorRecs)
			{
				QString processTextPre = (_context.defectDrawCfg.classIdNameMap.find(pairs.first) != _context.defectDrawCfg.classIdNameMap.end()) ?
					_context.defectDrawCfg.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
				auto currentIdCfg = _context.eliminationCfg[pairs.first];
				for (const auto& item : pairs.second)
				{
					if (currentIdCfg.isUsingArea)
					{
						errors.push_back(processTextPre + " : " +
							QString::number(item.area, 'f', 1)+ " "+ 
							QString::number(currentIdCfg.areaRange.first, 'f', 1)+" ~ " + 
							QString::number(currentIdCfg.areaRange.second, 'f', 1));
					}
					if (currentIdCfg.isUsingScore)
					{
						errors.push_back(processTextPre + " : " +
							QString::number(item.score, 'f', 1) + " " +
							QString::number(currentIdCfg.scoreRange.first, 'f', 1) + " ~ " +
							QString::number(currentIdCfg.scoreRange.second, 'f', 1));
					}
				}
			}
			_context.runTextConfig.extraTexts = errors;
			rw::imgPro::DefectDrawFunc::drawRunText(img, _context.runTextConfig);

			return img;
		}
	}
}
