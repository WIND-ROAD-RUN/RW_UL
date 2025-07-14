#include"rqw_HalconShapeModel.hpp"

#include <halconcpp/HalconCpp.h>

namespace rw
{
	namespace rqw
	{
        std::vector<HalconWidgetTemplateResult> HalconShapeModel::shape(
            const HalconShapeId& id, const HalconWidgetObject& rec)
        {
            return shape(id,&rec);
        }

        std::vector<HalconWidgetTemplateResult> HalconShapeModel::shape(const HalconShapeId& id,
            const HalconWidgetObject* rec)
        {
			return shape(id, rec, PainterConfig());
        }

        std::vector<HalconWidgetTemplateResult> HalconShapeModel::shape(const HalconShapeId& id,
	        const HalconWidgetObject* rec, const PainterConfig& config)
        {
            if (!rec)
            {
                return std::vector<HalconWidgetTemplateResult>();
            }

            // 检查输入对象是否有效
            if (!rec->has_value())
            {
                return std::vector<HalconWidgetTemplateResult>();
            }

            if (rec->type != HalconObjectType::Image)
            {
                return std::vector<HalconWidgetTemplateResult>();
            }

            // 获取图像对象
            HalconCpp::HObject* image = rec->value();

            // 执行模板匹配
            HalconCpp::HTuple row, column, angle, score;
            HalconCpp::FindShapeModel(*image, id, -0.39, 0.79, 0.5, 1, 0.5, "least_squares", 0, 0.9, &row, &column, &angle, &score);

            // 存储匹配结果
            std::vector<HalconWidgetTemplateResult> results;

            // 检查是否找到匹配
            if (row.TupleLength() > 0)
            {
                // 获取模板轮廓
                HalconCpp::HObject modelContours;
                HalconCpp::GetShapeModelContours(&modelContours, id, 1);

                // 遍历所有匹配结果
                for (int i = 0; i < row.TupleLength(); ++i)
                {
                    // 计算仿射变换矩阵
                    HalconCpp::HTuple homMat2D;
                    HalconCpp::VectorAngleToRigid(0, 0, 0, row[i], column[i], angle[i], &homMat2D);

                    // 仿射变换模板轮廓
                    HalconCpp::HObject transformedContours;
                    HalconCpp::AffineTransContourXld(modelContours, &transformedContours, homMat2D);

                    // 创建匹配结果对象
                    HalconWidgetTemplateResult result(new HalconCpp::HObject(transformedContours));
                    result.score = score[i].D(); // 设置匹配分数
                    result.painterConfig = config;

                    // 设置位置信息
                    result.row = row[i].D();      // 匹配中心的行坐标
                    result.column = column[i].D(); // 匹配中心的列坐标
                    result.angle = angle[i].D();   // 匹配的旋转角度

                    // 添加到结果列表
                    results.push_back(std::move(result));
                }
            }

            // 如果没有找到匹配，返回空结果列表
            return results;
        }
	}
}
