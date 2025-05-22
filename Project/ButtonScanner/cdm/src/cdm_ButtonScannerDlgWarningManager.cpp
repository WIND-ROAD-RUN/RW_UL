#include"cdm_ButtonScannerDlgWarningManager.h"

#include"oso_core.h"

namespace rw
{
	namespace cdm
	{
		ButtonScannerDlgWarningManager::ButtonScannerDlgWarningManager(const rw::oso::ObjectStoreAssembly& assembly)
		{
			auto isAccountAssembly = assembly.getName();
			if (isAccountAssembly != "$class$ButtonScannerDlgWarningManager$")
			{
				throw std::runtime_error("Assembly is not $class$ButtonScannerDlgWarningManager$");
			}
			auto cameraDisconnect1Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect1$"));
			if (!cameraDisconnect1Item) {
				throw std::runtime_error("$variable$cameraDisconnect1 is not found");
			}
			cameraDisconnect1 = cameraDisconnect1Item->getValueAsBool();
			auto cameraDisconnect2Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect2$"));
			if (!cameraDisconnect2Item) {
				throw std::runtime_error("$variable$cameraDisconnect2 is not found");
			}
			cameraDisconnect2 = cameraDisconnect2Item->getValueAsBool();
			auto cameraDisconnect3Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect3$"));
			if (!cameraDisconnect3Item) {
				throw std::runtime_error("$variable$cameraDisconnect3 is not found");
			}
			cameraDisconnect3 = cameraDisconnect3Item->getValueAsBool();
			auto cameraDisconnect4Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect4$"));
			if (!cameraDisconnect4Item) {
				throw std::runtime_error("$variable$cameraDisconnect4 is not found");
			}
			cameraDisconnect4 = cameraDisconnect4Item->getValueAsBool();
			auto workTrigger1Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger1$"));
			if (!workTrigger1Item) {
				throw std::runtime_error("$variable$workTrigger1 is not found");
			}
			workTrigger1 = workTrigger1Item->getValueAsBool();
			auto workTrigger2Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger2$"));
			if (!workTrigger2Item) {
				throw std::runtime_error("$variable$workTrigger2 is not found");
			}
			workTrigger2 = workTrigger2Item->getValueAsBool();
			auto workTrigger3Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger3$"));
			if (!workTrigger3Item) {
				throw std::runtime_error("$variable$workTrigger3 is not found");
			}
			workTrigger3 = workTrigger3Item->getValueAsBool();
			auto workTrigger4Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger4$"));
			if (!workTrigger4Item) {
				throw std::runtime_error("$variable$workTrigger4 is not found");
			}
			workTrigger4 = workTrigger4Item->getValueAsBool();
			auto airPressureItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$airPressure$"));
			if (!airPressureItem) {
				throw std::runtime_error("$variable$airPressure is not found");
			}
			airPressure = airPressureItem->getValueAsBool();
		}

		ButtonScannerDlgWarningManager::ButtonScannerDlgWarningManager(
			const ButtonScannerDlgWarningManager& buttonScannerDlgExposureTimeSet)
		{
			cameraDisconnect1 = buttonScannerDlgExposureTimeSet.cameraDisconnect1;
			cameraDisconnect2 = buttonScannerDlgExposureTimeSet.cameraDisconnect2;
			cameraDisconnect3 = buttonScannerDlgExposureTimeSet.cameraDisconnect3;
			cameraDisconnect4 = buttonScannerDlgExposureTimeSet.cameraDisconnect4;
			workTrigger1 = buttonScannerDlgExposureTimeSet.workTrigger1;
			workTrigger2 = buttonScannerDlgExposureTimeSet.workTrigger2;
			workTrigger3 = buttonScannerDlgExposureTimeSet.workTrigger3;
			workTrigger4 = buttonScannerDlgExposureTimeSet.workTrigger4;
			airPressure = buttonScannerDlgExposureTimeSet.airPressure;
		}

		ButtonScannerDlgWarningManager& ButtonScannerDlgWarningManager::operator=(
			const ButtonScannerDlgWarningManager& buttonScannerMainWindow)
		{
			if (this != &buttonScannerMainWindow) {
				cameraDisconnect1 = buttonScannerMainWindow.cameraDisconnect1;
				cameraDisconnect2 = buttonScannerMainWindow.cameraDisconnect2;
				cameraDisconnect3 = buttonScannerMainWindow.cameraDisconnect3;
				cameraDisconnect4 = buttonScannerMainWindow.cameraDisconnect4;
				workTrigger1 = buttonScannerMainWindow.workTrigger1;
				workTrigger2 = buttonScannerMainWindow.workTrigger2;
				workTrigger3 = buttonScannerMainWindow.workTrigger3;
				workTrigger4 = buttonScannerMainWindow.workTrigger4;
				airPressure = buttonScannerMainWindow.airPressure;
			}
			return *this;
		}

		ButtonScannerDlgWarningManager::operator oso::ObjectStoreAssembly() const
		{
			rw::oso::ObjectStoreAssembly assembly;
			assembly.setName("$class$ButtonScannerDlgWarningManager$");
			auto cameraDisconnect1Item = std::make_shared<oso::ObjectStoreItem>();
			cameraDisconnect1Item->setName("$variable$cameraDisconnect1$");
			cameraDisconnect1Item->setValueFromBool(cameraDisconnect1);
			assembly.addItem(cameraDisconnect1Item);
			auto cameraDisconnect2Item = std::make_shared<oso::ObjectStoreItem>();
			cameraDisconnect2Item->setName("$variable$cameraDisconnect2$");
			cameraDisconnect2Item->setValueFromBool(cameraDisconnect2);
			assembly.addItem(cameraDisconnect2Item);
			auto cameraDisconnect3Item = std::make_shared<oso::ObjectStoreItem>();
			cameraDisconnect3Item->setName("$variable$cameraDisconnect3$");
			cameraDisconnect3Item->setValueFromBool(cameraDisconnect3);
			assembly.addItem(cameraDisconnect3Item);
			auto cameraDisconnect4Item = std::make_shared<oso::ObjectStoreItem>();
			cameraDisconnect4Item->setName("$variable$cameraDisconnect4$");
			cameraDisconnect4Item->setValueFromBool(cameraDisconnect4);
			assembly.addItem(cameraDisconnect4Item);
			auto workTrigger1Item = std::make_shared<oso::ObjectStoreItem>();
			workTrigger1Item->setName("$variable$workTrigger1$");
			workTrigger1Item->setValueFromBool(workTrigger1);
			assembly.addItem(workTrigger1Item);
			auto workTrigger2Item = std::make_shared<oso::ObjectStoreItem>();
			workTrigger2Item->setName("$variable$workTrigger2$");
			workTrigger2Item->setValueFromBool(workTrigger2);
			assembly.addItem(workTrigger2Item);
			auto workTrigger3Item = std::make_shared<oso::ObjectStoreItem>();
			workTrigger3Item->setName("$variable$workTrigger3$");
			workTrigger3Item->setValueFromBool(workTrigger3);
			assembly.addItem(workTrigger3Item);
			auto workTrigger4Item = std::make_shared<oso::ObjectStoreItem>();
			workTrigger4Item->setName("$variable$workTrigger4$");
			workTrigger4Item->setValueFromBool(workTrigger4);
			assembly.addItem(workTrigger4Item);
			auto airPressureitem = std::make_shared<oso::ObjectStoreItem>();
			airPressureitem->setName("$variable$airPressure$");
			airPressureitem->setValueFromBool(airPressure);
			assembly.addItem(airPressureitem);
			return assembly;
		}

		bool ButtonScannerDlgWarningManager::operator==(const ButtonScannerDlgWarningManager& account) const
		{
			return cameraDisconnect1 == account.cameraDisconnect1 &&
				cameraDisconnect2 == account.cameraDisconnect2 &&
				cameraDisconnect3 == account.cameraDisconnect3 &&
				cameraDisconnect4 == account.cameraDisconnect4 &&
				workTrigger1 == account.workTrigger1 &&
				workTrigger2 == account.workTrigger2 &&
				workTrigger3 == account.workTrigger3 &&
				workTrigger4 == account.workTrigger4 &&
				airPressure == account.airPressure;
		}

		bool ButtonScannerDlgWarningManager::operator!=(const ButtonScannerDlgWarningManager& account) const
		{
			return !(*this == account);
		}
	}
}
