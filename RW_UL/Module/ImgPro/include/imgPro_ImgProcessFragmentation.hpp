#pragma once

#include <shared_mutex>

#include"imgPro_ImageProcess.hpp"

namespace rw
{
	namespace imgPro
	{
		class ImageProcessFragmentation
		{
		public:
			explicit ImageProcessFragmentation(ImageProcess * imageProcess);
			~ImageProcessFragmentation();
		private:
			std::shared_ptr<ImageProcess> _imageProcess;
		public:
			const ImageProcessContext& getContext() const;

			ImageProcessContext& context();
		public:
			std::unique_ptr<rw::ModelEngine>& getModelEngine();
		public:
			RunTime getProcessImgTime();
			RunTime getOperatorTime();
		public:
			ProcessResult processImg(const cv::Mat& mat);
			ProcessResultIndexMap getIndex(const ProcessResult& processResult);
			EliminationInfo getEliminationInfo(const ProcessResult& processResult, const ProcessResultIndexMap& indexMap, const rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap& configs);
			DefectResultInfo getDefectResultInfo(const EliminationInfo& eliminationInfo, const rw::imgPro::DefectResultInfoFunc::ClassIdWithConfigMap& configs);
		private:
			std::vector<cv::Mat> _matFragmentation{};
			mutable std::shared_mutex _mutex; 
		public:
			cv::Mat collageFragmentationMat();
			void pushFragmentationMat(const cv::Mat & mat);
		public:
			void operator()();
			QImage getMaskImg(const cv::Mat& mat);
			static QImage getMaskImg(const cv::Mat& mat,
				const DefectResultInfo& defectResultInfo,
				const ProcessResult& processResult,
				ImageProcessContext& context,
				RunTime operatorTime,
				RunTime processImgTime
			);
			static void getMaskImg(QImage& img,
				const DefectResultInfo& defectResultInfo,
				const ProcessResult& processResult,
				ImageProcessContext& context,
				RunTime operatorTime,
				RunTime processImgTime
			);
		public:
			const ProcessResultIndexMap& getProcessResultIndexMap() const;
			const ProcessResult& getProcessResult() const;
			const EliminationInfo& getEliminationInfo() const;
			const DefectResultInfo& getDefectResultInfo() const;

			ProcessResultIndexMap& processResultIndexMap();
			ProcessResult& processResult();
			EliminationInfo& eliminationInfo();
			DefectResultInfo& defectResultInfo();
		};
	}
}