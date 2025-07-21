#include "osoc_MVSCameraConfig.h"

#include"oso_core.h"

#include<cassert>

namespace rw {
	namespace oso {
		ObjectStoreAssembly
			CameraAttributesBasic::toObjectStoreAssembly
			(const CameraAttributesBasic& cameraAttributesBasic)
		{
			ObjectStoreAssembly assembly;
			assembly.setName("CameraAttributesBasic");

			ObjectStoreItem exposureTime;
			exposureTime.setName("ExposureTime");
			exposureTime.setValueFromLong(cameraAttributesBasic.exposureTime);
			assembly.addItem(makeObjectStoreItemSharedPtr(exposureTime));

			ObjectStoreItem gain;
			gain.setName("Gain");
			gain.setValueFromInt(cameraAttributesBasic.gain);
			assembly.addItem(makeObjectStoreItemSharedPtr(gain));

			return assembly;
		}

		CameraAttributesBasic
			CameraAttributesBasic::toCameraAttributesBasic
			(const ObjectStoreAssembly& assembly)
		{
			CameraAttributesBasic cameraAttributesBasic;

			for (const auto& item : assembly.getItems())
			{
				std::shared_ptr<ObjectStoreItem> nItem = std::dynamic_pointer_cast<ObjectStoreItem>(item);
				if (nItem->getName() == "ExposureTime")
				{
					cameraAttributesBasic.exposureTime = nItem->getValueAsLong();
				}
				else if (nItem->getName() == "Gain")
				{
					cameraAttributesBasic.gain = nItem->getValueAsInt();
				}
			}

			return cameraAttributesBasic;
		}

		CameraAttributesBasic::operator ObjectStoreAssembly() const
		{
			return toObjectStoreAssembly(*this);
		}

		CameraAttributesBasic::CameraAttributesBasic(const ObjectStoreAssembly& assembly)
		{
			*this = toCameraAttributesBasic(assembly);
		}

		CameraAttributesBasic::CameraAttributesBasic(std::shared_ptr<ObjectStoreAssembly> assembly)
		{
			*this = toCameraAttributesBasic(*assembly);
		}

		ObjectStoreAssembly
			CameraAttributesAdvanced::toObjectStoreAssembly
			(const CameraAttributesAdvanced& cameraAttributesBasic)
		{
			assert(cameraAttributesBasic.mode != CameraTriggerMode::Undefined);

			ObjectStoreItem mode;
			mode.setName("CameraTriggerMode");
			switch (cameraAttributesBasic.mode)
			{
			case CameraTriggerMode::Software:
				mode.setValueFromString("Software");
				break;
			case CameraTriggerMode::Hardware:
				mode.setValueFromString("Hardware");
				break;
			}

			ObjectStoreAssembly assembly;
			assembly.setName("CameraAttributesAdvanced");
			assembly.addItem(makeObjectStoreItemSharedPtr(mode));

			return assembly;
		}

		CameraAttributesAdvanced
			CameraAttributesAdvanced::toCameraAttributesAdvanced
			(const ObjectStoreAssembly& assembly)
		{
			CameraAttributesAdvanced cameraAttributesAdvanced;

			for (const auto& item : assembly.getItems())
			{
				auto nItem = ObjectStoreCoreToItem(item);
				if (nItem->getName() == "CameraTriggerMode")
				{
					if (nItem->getValueAsString() == "Software")
					{
						cameraAttributesAdvanced.mode = CameraTriggerMode::Software;
					}
					else if (nItem->getValueAsString() == "Hardware")
					{
						cameraAttributesAdvanced.mode = CameraTriggerMode::Hardware;
					}
				}
			}

			return cameraAttributesAdvanced;
		}

		CameraAttributesAdvanced::CameraAttributesAdvanced(const ObjectStoreAssembly& assembly)
		{
			*this = toCameraAttributesAdvanced(assembly);
		}

		CameraAttributesAdvanced::CameraAttributesAdvanced(std::shared_ptr<ObjectStoreAssembly> assembly)
		{
			*this = toCameraAttributesAdvanced(*assembly);
		}

		CameraAttributesAdvanced::operator ObjectStoreAssembly() const
		{
			ObjectStoreAssembly assembly;
			assembly.setName("CameraAttributesAdvanced");

			ObjectStoreItem triggerMode;
			triggerMode.setName("CameraTriggerMode");
			switch (mode)
			{
			case CameraTriggerMode::Software:
				triggerMode.setValueFromString("Software");
				break;
			case CameraTriggerMode::Hardware:
				triggerMode.setValueFromString("Hardware");
				break;
			}

			assembly.addItem(makeObjectStoreItemSharedPtr(triggerMode));

			return assembly;
		}
	}
}