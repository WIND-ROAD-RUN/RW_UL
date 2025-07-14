#pragma once

#include"rqw_HalconUtilty.hpp"
#include"rqw_HalconWidgetDisObject.hpp"

namespace rw {
	namespace rqw
	{
		class HalconShapeModel {
		public:
			static std::vector<HalconWidgetTemplateResult> shape(const HalconShapeId& id, const HalconWidgetObject& rec);
			static std::vector<HalconWidgetTemplateResult> shape(const HalconShapeId& id, const HalconWidgetObject * rec);

			static std::vector<HalconWidgetTemplateResult> shape(const HalconShapeId& id, const HalconWidgetObject* rec,const PainterConfig&config);
		};
	}
}