#pragma once
namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
	}
}

namespace rw
{
	namespace cdm {
		class ButtonScannerProduceLineSet
		{
		public:
			ButtonScannerProduceLineSet() = default;
			~ButtonScannerProduceLineSet() = default;

			ButtonScannerProduceLineSet(const rw::oso::ObjectStoreAssembly& assembly);
			ButtonScannerProduceLineSet(const ButtonScannerProduceLineSet& buttonScannerMainWindow);
			ButtonScannerProduceLineSet& operator=(const ButtonScannerProduceLineSet& buttonScannerMainWindow);

		public:
			bool blowingEnable1{ false };
			bool blowingEnable2{ false };
			bool blowingEnable3{ false };
			bool blowingEnable4{ false };

			double  blowDistance1{ 0 };
			double  blowDistance2{ 0 };
			double  blowDistance3{ 0 };
			double  blowDistance4{ 0 };

			double  blowTime1{ 0 };
			double  blowTime2{ 0 };
			double  blowTime3{ 0 };
			double  blowTime4{ 0 };

			double  pixelEquivalent1{ 0 };
			double  pixelEquivalent2{ 0 };
			double  pixelEquivalent3{ 0 };
			double  pixelEquivalent4{ 0 };

			double limit1{ 0 };
			double limit2{ 0 };
			double limit3{ 0 };
			double limit4{ 0 };

			double minBrightness{ 0 };
			double maxBrightness{ 0 };

			bool powerOn{ false };
			bool none{ false };
			bool run{ false };
			bool alarm{ false };

			bool workstationProtection12{ false };
			bool workstationProtection34{ false };
			bool debugMode{ false };

			double motorSpeed{ 0 };
			double beltReductionRatio{ 0 };
			double accelerationAndDeceleration{ 0 };
			double codeWheel{ 0 };
			double pulseFactor{ 0 };

		public:
			operator rw::oso::ObjectStoreAssembly() const;
			bool operator==(const ButtonScannerProduceLineSet& account) const;
			bool operator!=(const ButtonScannerProduceLineSet& account) const;
		};
	}
}