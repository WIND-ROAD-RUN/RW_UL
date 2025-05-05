#include"cdm_ButtonScannerDlgHideScoreSet.h"

#include"oso_core.h"

namespace rw
{
	namespace cdm
	{
		DlgHideScoreSet::DlgHideScoreSet(const rw::oso::ObjectStoreAssembly& assembly)
		{
			auto isAccountAssembly = assembly.getName();
			if (isAccountAssembly != "$class$DlgHideScoreSet$") {
				throw std::runtime_error("Assembly is not $class$DlgHideScoreSet$");
			}
			auto outsideDiameterScoreItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterScore$"));
			if (!outsideDiameterScoreItem) {
				throw std::runtime_error("$variable$outsideDiameterScore is not found");
			}
			outsideDiameterScore = outsideDiameterScoreItem->getValueAsDouble();
			auto forAndAgainstScoreItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$forAndAgainstScore$"));
			if (!forAndAgainstScoreItem) {
				throw std::runtime_error("$variable$forAndAgainstScore is not found");
			}
			forAndAgainstScore = forAndAgainstScoreItem->getValueAsDouble();
		}

		DlgHideScoreSet::DlgHideScoreSet(const DlgHideScoreSet& buttonScannerMainWindow)
		{
			outsideDiameterScore = buttonScannerMainWindow.outsideDiameterScore;
			forAndAgainstScore = buttonScannerMainWindow.forAndAgainstScore;
		}

		DlgHideScoreSet& DlgHideScoreSet::operator=(const DlgHideScoreSet& buttonScannerMainWindow)
		{
			if (this != &buttonScannerMainWindow) {
				outsideDiameterScore = buttonScannerMainWindow.outsideDiameterScore;
				forAndAgainstScore = buttonScannerMainWindow.forAndAgainstScore;
			}
			return *this;
		}

		DlgHideScoreSet::operator oso::ObjectStoreAssembly() const
		{
			rw::oso::ObjectStoreAssembly assembly;
			assembly.setName("$class$DlgHideScoreSet$");

			auto outsideDiameterScoreItem = std::make_shared<oso::ObjectStoreItem>();
			outsideDiameterScoreItem->setName("$variable$outsideDiameterScore$");
			outsideDiameterScoreItem->setValueFromDouble(outsideDiameterScore);
			assembly.addItem(outsideDiameterScoreItem);

			auto forAndAgainstScoreItem = std::make_shared<oso::ObjectStoreItem>();
			forAndAgainstScoreItem->setName("$variable$forAndAgainstScore$");
			forAndAgainstScoreItem->setValueFromDouble(forAndAgainstScore);
			assembly.addItem(forAndAgainstScoreItem);
			return assembly;
		}

		bool DlgHideScoreSet::operator==(const DlgHideScoreSet& account) const
		{
			return outsideDiameterScore == account.outsideDiameterScore &&
				forAndAgainstScore == account.forAndAgainstScore;
		}

		bool DlgHideScoreSet::operator!=(const DlgHideScoreSet& account) const
		{
			return !(*this == account);
		}
	}
}