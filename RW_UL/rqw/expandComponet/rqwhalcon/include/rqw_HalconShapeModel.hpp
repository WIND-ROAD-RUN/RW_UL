#pragma once

#include"rqw_HalconUtilty.hpp"
#include"rqw_HalconWidgetDisObject.hpp"

namespace rw {
	namespace rqw
	{
		class HalconShapeModel {
		public:
			static HalconWidgetObject shape(const HalconShapeId& id,const HalconWidgetObject& rec);

		};
	}
}