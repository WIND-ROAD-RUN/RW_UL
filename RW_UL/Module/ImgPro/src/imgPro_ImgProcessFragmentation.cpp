#include"imgPro_ImgProcessFragmentation.hpp"

namespace rw
{
	namespace imgPro
	{
		ImageProcessFragmentation::ImageProcessFragmentation(ImageProcess& imageProcess)
			:_imageProcess(imageProcess)
		{
		}

		ImageProcessFragmentation::~ImageProcessFragmentation()
		{

		}

		const ImageProcessContext& ImageProcessFragmentation::getContext() const
		{
			return _imageProcess.getContext();
		}

		ImageProcessContext& ImageProcessFragmentation::context()
		{
			return _imageProcess.context();
		}

		std::unique_ptr<rw::ModelEngine>& ImageProcessFragmentation::getModelEngine()
		{
			return _imageProcess.getModelEngine();
		}

		RunTime ImageProcessFragmentation::getProcessImgTime()
		{
			return _imageProcess.getProcessImgTime();
		}

		RunTime ImageProcessFragmentation::getOperatorTime()
		{
			return _imageProcess.getOperatorTime();
		}

		ProcessResult ImageProcessFragmentation::processImg(const cv::Mat& mat)
		{
			return _imageProcess.processImg(mat);
		}

		ProcessResultIndexMap ImageProcessFragmentation::getIndex(const ProcessResult& processResult)
		{
			return _imageProcess.getIndex(processResult);
		}

		EliminationInfo ImageProcessFragmentation::getEliminationInfo(const ProcessResult& processResult,
			const ProcessResultIndexMap& indexMap, const rw::imgPro::EliminationInfoFunc::ClassIdWithConfigMap& configs)
		{
			return _imageProcess.getEliminationInfo(processResult, indexMap, configs);
		}

		DefectResultInfo ImageProcessFragmentation::getDefectResultInfo(const EliminationInfo& eliminationInfo,
			const rw::imgPro::DefectResultInfoFunc::ClassIdWithConfigMap& configs)
		{
			return _imageProcess.getDefectResultInfo(eliminationInfo, configs);
		}

		cv::Mat ImageProcessFragmentation::collageFragmentationMat()
		{
			std::vector<cv::Mat> localFragmentation;
			{
				std::unique_lock lock(_mutex);
				if (_matFragmentation.empty())
				{
					return cv::Mat();
				}
				localFragmentation = std::move(_matFragmentation);
				_matFragmentation.clear();
			}

			int totalHeight = 0;
			int maxWidth = 0;
			for (const auto& img : localFragmentation)
			{
				totalHeight += img.rows;
				maxWidth = std::max(maxWidth, img.cols);
			}

			cv::Mat collageImage(totalHeight, maxWidth, localFragmentation[0].type(), cv::Scalar(0, 0, 0));

			int currentY = 0;
			for (const auto& img : localFragmentation)
			{
				img.copyTo(collageImage(cv::Rect(0, currentY, img.cols, img.rows)));
				currentY += img.rows;
			}
			return collageImage;
		}

		void ImageProcessFragmentation::pushFragmentationMat(const cv::Mat& mat)
		{
			std::unique_lock lock(_mutex);
			_matFragmentation.push_back(mat.clone());
		}

		void ImageProcessFragmentation::operator()()
		{
			_imageProcess(collageFragmentationMat());
		}

		QImage ImageProcessFragmentation::getMaskImg(const cv::Mat& mat)
		{
			return _imageProcess.getMaskImg(mat);
		}

		QImage ImageProcessFragmentation::getMaskImg(const cv::Mat& mat, const DefectResultInfo& defectResultInfo,
			const ProcessResult& processResult, ImageProcessContext& context, RunTime operatorTime,
			RunTime processImgTime)
		{
			return ImageProcess::getMaskImg(mat, defectResultInfo, processResult, context, operatorTime, processImgTime);
		}

		void ImageProcessFragmentation::getMaskImg(QImage& img, const DefectResultInfo& defectResultInfo,
			const ProcessResult& processResult, ImageProcessContext& context, RunTime operatorTime,
			RunTime processImgTime)
		{
			ImageProcess::getMaskImg(img, defectResultInfo, processResult, context, operatorTime, processImgTime);
		}

		const ProcessResultIndexMap& ImageProcessFragmentation::getProcessResultIndexMap() const
		{
			return _imageProcess.getProcessResultIndexMap();
		}

		const ProcessResult& ImageProcessFragmentation::getProcessResult() const
		{
			return _imageProcess.getProcessResult();
		}

		const EliminationInfo& ImageProcessFragmentation::getEliminationInfo() const
		{
			return _imageProcess.getEliminationInfo();
		}

		const DefectResultInfo& ImageProcessFragmentation::getDefectResultInfo() const
		{
			return _imageProcess.getDefectResultInfo();
		}

		ProcessResultIndexMap& ImageProcessFragmentation::processResultIndexMap()
		{
			return _imageProcess.processResultIndexMap();
		}

		ProcessResult& ImageProcessFragmentation::processResult()
		{
			return _imageProcess.processResult();
		}

		EliminationInfo& ImageProcessFragmentation::eliminationInfo()
		{
			return _imageProcess.eliminationInfo();
		}

		DefectResultInfo& ImageProcessFragmentation::defectResultInfo()
		{
			return _imageProcess.defectResultInfo();
		}
	}
}
