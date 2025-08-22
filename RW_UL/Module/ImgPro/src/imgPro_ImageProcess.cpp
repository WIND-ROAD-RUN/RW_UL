#include"imgPro_ImageProcess.hpp"

#include "imgPro_IndexFunc.hpp"

#include <chrono>

#include "rqw_ImgConvert.hpp"

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

			_context._processImgTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			_context._processResult = processResult;
			return _context._processResult;
		}

		ProcessResultIndexMap ImageProcess::getIndex(const ProcessResult& processResult)
		{
			auto indexMap = rw::imgPro::IndexFunc::getIndex(processResult);

			auto& indexGetContext = _context.indexGetContext;

			if (indexGetContext.removeIndicesIf)
			{
				indexGetContext.removedIndices =
					rw::imgPro::IndexFunc::removeIndicesIf(
						indexMap, indexGetContext.removeIndicesIf);
			}
			if (indexGetContext.removeIndicesIfByInfo)
			{
				indexGetContext.removedIndicesByInfo =
					rw::imgPro::IndexFunc::removeIndicesIfByInfo(
						indexMap, _context._processResult, indexGetContext.removeIndicesIfByInfo);
			}

			_processResultIndexMap = indexMap;
			return _processResultIndexMap;
		}

		EliminationInfo ImageProcess::getEliminationInfo(const ProcessResult& processResult,
			const ProcessResultIndexMap& indexMap,
			const rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap& configs)
		{
			auto eliminationInfo =
				rw::imgPro::EliminationInfoFunc::getEliminationInfo(
					processResult,
					indexMap,
					configs,
					_context.eliminationInfoGetContext.getEliminationItemFuncSpecialOperator,
					_context.eliminationInfoGetContext.getEliminationItemPostOperator
				);
			_eliminationInfo = eliminationInfo;
			return _eliminationInfo;
		}

		DefectResultInfo ImageProcess::getDefectResultInfo(
			const ProcessResult& processResult,
			const EliminationInfo& eliminationInfo,
			const rw::imgPro::DefectResultInfoFunc::ClassIdWithConfigMap& configs,
			const rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap& eliConfigs
		)
		{
			auto defectResultInfo =
				rw::imgPro::DefectResultInfoFunc::getDefectResultInfo(
					processResult,
					eliConfigs,
					eliminationInfo,
					configs,
					_context.defectResultGetContext.getDefectResultExtraOperate,
					_context.defectResultGetContext.getDefectResultExtraOperateDisable,
					_context.defectResultGetContext.getDefectResultExtraOperateWithFullInfo,
					_context.defectResultGetContext.getDefectResultExtraPostOperate
				);
			_defectResultInfo = defectResultInfo;
			return _defectResultInfo;
		}

		void ImageProcess::operator()(const cv::Mat& mat)
		{
			if (_context.imageProcessPrepare)
			{
				_context.imageProcessPrepare(_context);
			}
			auto start = std::chrono::high_resolution_clock::now();

			processImg(mat);
			getIndex(_context._processResult);
			getEliminationInfo(_context._processResult, _processResultIndexMap, _context.eliminationCfg);
			getDefectResultInfo(_context._processResult,_eliminationInfo, _context.defectCfg, _context.eliminationCfg);

			auto end = std::chrono::high_resolution_clock::now();

			_context._operatorTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		}

		QImage ImageProcess::getMaskImg(const cv::Mat& mat)
		{
			return getMaskImg(mat, _defectResultInfo, _context._processResult, _context, _context._operatorTime, _context._processImgTime);
		}

		QImage ImageProcess::getMaskImg(const cv::Mat& mat, const DefectResultInfo& defectResultInfo,
			const ProcessResult& processResult, ImageProcessContext& context, RunTime operatorTime,
			RunTime processImgTime)
		{
			auto img = rw::CvMatToQImage(mat);
			getMaskImg(img, defectResultInfo, processResult, context, operatorTime, processImgTime);
			return img;
		}

		void ImageProcess::getMaskImg(QImage& img, const DefectResultInfo& defectResultInfo,
			const ProcessResult& processResult, ImageProcessContext& context, RunTime operatorTime,
			RunTime processImgTime)
		{
			auto  &defectDrawCfg = context.defectDrawCfg;
			rw::imgPro::DefectDrawFunc::drawDefectRecs(img, defectResultInfo, processResult, defectDrawCfg, context.defectDrawFuncContext);
			context.runTextCfg.operatorTimeText = QString::number(operatorTime) + " ms";
			context.runTextCfg.processImgTimeText = QString::number(processImgTime) + " ms";
			auto& errorRecs = defectResultInfo.defects;
			QVector<QString> errors;
			for (auto& pairs : errorRecs)
			{
				if (defectDrawCfg.classIdIgnoreDrawSet.find(pairs.first) != defectDrawCfg.classIdIgnoreDrawSet.end())
				{
					continue;
				}
				QString processTextPre = (defectDrawCfg.classIdNameMap.find(pairs.first) != defectDrawCfg.classIdNameMap.end()) ?
					defectDrawCfg.classIdNameMap.at(pairs.first) : QString::number(pairs.first);
				auto currentIdCfg = context.eliminationCfg[pairs.first];
				for (const auto& item : pairs.second)
				{
					if (currentIdCfg.isUsingArea)
					{
						errors.push_back(processTextPre + " : " +
							QString::number(item.area, 'f', defectDrawCfg.areaDisPrecision) + " " +
							QString::number(currentIdCfg.areaRange.first, 'f', defectDrawCfg.areaDisPrecision) + " ~ " +
							QString::number(currentIdCfg.areaRange.second, 'f', defectDrawCfg.areaDisPrecision));
					}
					if (currentIdCfg.isUsingScore)
					{
						errors.push_back(processTextPre + " : " +
							QString::number(item.score, 'f', defectDrawCfg.scoreDisPrecision) + " " +
							QString::number(currentIdCfg.scoreRange.first, 'f', defectDrawCfg.scoreDisPrecision) + " ~ " +
							QString::number(currentIdCfg.scoreRange.second, 'f', defectDrawCfg.scoreDisPrecision));
					}
				}
			}
			context.runTextCfg.extraTexts = errors;
			rw::imgPro::DefectDrawFunc::drawRunText(img, context.runTextCfg);

			if (context.defectDrawFuncContext.postOperateFunc)
			{
				context.defectDrawFuncContext.postOperateFunc(img, context);
			}
		}
	}
}