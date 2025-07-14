#include"rqw_HalconShapeModel.hpp"

#include <halconcpp/HalconCpp.h>

namespace rw
{
	namespace rqw
	{
		HalconWidgetObject HalconShapeModel::shape(const HalconShapeId& id, const HalconWidgetObject& rec)
		{
			if (!rec.has_value())
			{
				throw std::runtime_error("The provided HalconWidgetDisObject does not contain a valid HObject.");
			}

			if (rec.type!=HalconObjectType::Image)
			{
				throw std::runtime_error("The provided HalconWidgetDisObject is not of type Image.");
			}

            // 获取图像对象
            HalconCpp::HObject* image = rec.value();

            // 执行模板匹配
            HalconCpp::HTuple row, column, angle, score;
            HalconCpp::FindShapeModel(*image, id, -0.39, 0.79, 0.5, 1, 0.5, "least_squares", 0, 0.9, &row, &column, &angle, &score);

            // 检查是否找到匹配
            if (row.TupleLength() > 0)
            {
                // 获取模板轮廓
                HalconCpp::HObject modelContours, transformedContours;
                HalconCpp::GetShapeModelContours(&modelContours, id, 1);

                // 计算仿射变换矩阵并生成匹配结果
                HalconCpp::HTuple homMat2D;
                HalconCpp::VectorAngleToRigid(0, 0, 0, row[0], column[0], angle[0], &homMat2D);
                HalconCpp::AffineTransContourXld(modelContours, &transformedContours, homMat2D);

                // 返回匹配结果作为 HalconWidgetDisObject
                return HalconWidgetObject(new HalconCpp::HObject(transformedContours));
            }
            else
            {
                // 如果没有找到匹配，抛出异常或返回空对象
                throw std::runtime_error("No match found for the given shape model.");
            }
		}
	}
}
