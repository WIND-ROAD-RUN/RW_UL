#include "cdm_ButtonScannerDlgExposureTimeSet.h"

#include"oso_core.h"

namespace rw {
	namespace cdm {
		ButtonScannerDlgExposureTimeSet::ButtonScannerDlgExposureTimeSet(const rw::oso::ObjectStoreAssembly& assembly)
		{
			auto isAccountAssembly = assembly.getName();
			if (isAccountAssembly != "$class$ButtonScannerDlgExposureTimeSet$")
			{
				throw std::runtime_error("Assembly is not $class$ButtonScannerDlgExposureTimeSet$");
			}
			auto expousureTimeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$expousureTime$"));
			if (!expousureTimeItem) {
				throw std::runtime_error("$variable$expousureTime is not found");
			}
			expousureTime = expousureTimeItem->getValueAsDouble();
		}
		ButtonScannerDlgExposureTimeSet::ButtonScannerDlgExposureTimeSet(const ButtonScannerDlgExposureTimeSet& buttonScannerDlgExposureTimeSet)
		{
			expousureTime = buttonScannerDlgExposureTimeSet.expousureTime;
		}
		ButtonScannerDlgExposureTimeSet& ButtonScannerDlgExposureTimeSet::operator=(const ButtonScannerDlgExposureTimeSet& buttonScannerMainWindow)
		{
			if (this != &buttonScannerMainWindow) {
				expousureTime = buttonScannerMainWindow.expousureTime;
			}
			return *this;
		}
		ButtonScannerDlgExposureTimeSet::operator rw::oso::ObjectStoreAssembly() const
		{
			rw::oso::ObjectStoreAssembly assembly;
			assembly.setName("$class$ButtonScannerDlgExposureTimeSet$");

			auto expousureTimeItem = std::make_shared<oso::ObjectStoreItem>();
			expousureTimeItem->setName("$variable$expousureTime$");
			expousureTimeItem->setValueFromDouble(expousureTime);
			assembly.addItem(expousureTimeItem);

			return assembly;
		}

		bool ButtonScannerDlgExposureTimeSet::operator==(const ButtonScannerDlgExposureTimeSet& account) const
		{
			return expousureTime == account.expousureTime;
		}

		bool ButtonScannerDlgExposureTimeSet::operator!=(const ButtonScannerDlgExposureTimeSet& account) const
		{
			return !(*this == account);
		}
	}
}