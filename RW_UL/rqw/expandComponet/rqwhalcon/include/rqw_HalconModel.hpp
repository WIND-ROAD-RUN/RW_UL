#pragma once

#include"rqw_HalconUtilty.hpp"
#include"rqw_HalconWidgetDisObject.hpp"

namespace rw {
	namespace rqw
	{
		class HalconShapeModel {
		public:
			static HalconShapeId create(const HalconWidgetImg& img, const HalconWidgetObject& rec);
			static HalconShapeId create(const HalconWidgetObject* img, const HalconWidgetObject* rec);
			static HalconShapeId create(const HalconWidgetObject& img, const HalconWidgetObject* rec);
			static HalconShapeId create(const HalconWidgetObject* img, const HalconWidgetObject& rec);
		public:
			static std::vector<HalconWidgetTemplateResult> shape(const HalconShapeId& id, const HalconWidgetObject& rec);
			static std::vector<HalconWidgetTemplateResult> shape(const HalconShapeId& id, const HalconWidgetObject * rec);
			static std::vector<HalconWidgetTemplateResult> shape(const HalconShapeId& id, const HalconWidgetObject* rec,const PainterConfig&config);
		public:
			static void saveModel(const HalconShapeId& id, const std::string& filePath);
			static HalconShapeId readModel(const std::string& filePath);
		};
	}
}