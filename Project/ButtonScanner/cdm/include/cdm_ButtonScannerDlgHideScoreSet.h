#pragma once

#include <string>

namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
	}
}

namespace rw
{
	namespace cdm
	{
		class DlgHideScoreSet
		{
		public:
			DlgHideScoreSet() = default;
			~DlgHideScoreSet() = default;

			DlgHideScoreSet(const rw::oso::ObjectStoreAssembly& assembly);
			DlgHideScoreSet(const DlgHideScoreSet& buttonScannerMainWindow);

			DlgHideScoreSet& operator=(const DlgHideScoreSet& buttonScannerMainWindow);
			operator rw::oso::ObjectStoreAssembly() const;
			bool operator==(const DlgHideScoreSet& account) const;
			bool operator!=(const DlgHideScoreSet& account) const;
		public:
			double outsideDiameterScore{ 0 };
			double forAndAgainstScore{ 0 };
		};
	}
}
