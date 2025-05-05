#pragma once

#include <string>

namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
	}
}

namespace rw
{
	namespace cdm {
		class ButtonScannerMainWindow
		{
		public:
			ButtonScannerMainWindow() = default;
			~ButtonScannerMainWindow() = default;

			ButtonScannerMainWindow(const rw::oso::ObjectStoreAssembly& assembly);
			ButtonScannerMainWindow(const ButtonScannerMainWindow& buttonScannerMainWindow);
			ButtonScannerMainWindow& operator=(const ButtonScannerMainWindow& buttonScannerMainWindow);
		public:
			unsigned long totalProduction{ 0 };
			unsigned long totalWaste{ 0 };
			double passRate{ 0 };
		public:
			bool isDebugMode{ false };
			bool isTakePictures{ false };
			bool isEliminating{ false };
			bool scrappingRate{ false };
		public:
			bool upLight{ false };
			bool downLight{ false };
			bool sideLight{ false };
			double speed{ 0 };
			double beltSpeed{ 0 };
		public:
			bool isDefect{ false };
			bool isPositive{ false };

		public:
			operator rw::oso::ObjectStoreAssembly() const;
			bool operator==(const ButtonScannerMainWindow& account) const;
			bool operator!=(const ButtonScannerMainWindow& account) const;
		};
	}
}