#pragma once

namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
	}
}

namespace rw
{
	namespace cdm {
		class ButtonScannerDlgWarningManager
		{
		public:
			ButtonScannerDlgWarningManager() = default;
			~ButtonScannerDlgWarningManager() = default;

			ButtonScannerDlgWarningManager(const rw::oso::ObjectStoreAssembly& assembly);
			ButtonScannerDlgWarningManager(const ButtonScannerDlgWarningManager& buttonScannerDlgExposureTimeSet);

			ButtonScannerDlgWarningManager& operator=(const ButtonScannerDlgWarningManager& buttonScannerMainWindow);
			operator rw::oso::ObjectStoreAssembly() const;
			bool operator==(const ButtonScannerDlgWarningManager& account) const;
			bool operator!=(const ButtonScannerDlgWarningManager& account) const;

		public:
			bool cameraDisconnect1{true};
			bool cameraDisconnect2{ true };
			bool cameraDisconnect3{ true };
			bool cameraDisconnect4{ true };
		public:
			bool workTrigger1{true};
			bool workTrigger2{ true };
			bool workTrigger3{ true };
			bool workTrigger4{ true };
		public:
			bool airPressure{true};
		};
	}
}