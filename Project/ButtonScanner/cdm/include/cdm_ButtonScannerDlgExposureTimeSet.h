#pragma once

namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
	}
}

namespace rw
{
	namespace cdm {
		class ButtonScannerDlgExposureTimeSet
		{
		public:
			ButtonScannerDlgExposureTimeSet() = default;
			~ButtonScannerDlgExposureTimeSet() = default;

			ButtonScannerDlgExposureTimeSet(const rw::oso::ObjectStoreAssembly& assembly);
			ButtonScannerDlgExposureTimeSet(const ButtonScannerDlgExposureTimeSet& buttonScannerDlgExposureTimeSet);

			ButtonScannerDlgExposureTimeSet& operator=(const ButtonScannerDlgExposureTimeSet& buttonScannerMainWindow);
			operator rw::oso::ObjectStoreAssembly() const;
			bool operator==(const ButtonScannerDlgExposureTimeSet& account) const;
			bool operator!=(const ButtonScannerDlgExposureTimeSet& account) const;

		public:
			size_t expousureTime{ 1000 };
		};
	}
}