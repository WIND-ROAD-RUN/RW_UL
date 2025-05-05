#pragma once
namespace rw {
	namespace oso {
		class ObjectStoreAssembly;
	}
}
namespace rw
{
	namespace cdm {
		class ButtonScannerDlgProductSet
		{
		public:
			ButtonScannerDlgProductSet() = default;
			~ButtonScannerDlgProductSet() = default;

			ButtonScannerDlgProductSet(const rw::oso::ObjectStoreAssembly& assembly);
			ButtonScannerDlgProductSet(const ButtonScannerDlgProductSet& buttonScannerMainWindow);

			ButtonScannerDlgProductSet& operator=(const ButtonScannerDlgProductSet& buttonScannerMainWindow);
			operator rw::oso::ObjectStoreAssembly() const;
			bool operator==(const ButtonScannerDlgProductSet& account) const;
			bool operator!=(const ButtonScannerDlgProductSet& account) const;

		public:
			bool outsideDiameterEnable{ false };
			double outsideDiameterValue{ 0 };
			double outsideDiameterDeviation{ 0 };

			double photography{ 0 };
			double blowTime{ 0 };

			bool edgeDamageEnable{ false };
			double edgeDamageSimilarity{ false };

			bool shieldingRangeEnable{ false };
			double outerRadius{ 0 };
			double innerRadius{ 0 };

			bool poreEnable{ false };
			double poreEnableScore{ 0 };//

			bool paintEnable{ false };
			double paintEnableScore{ 0 };//

			bool holesCountEnable{ false };
			double holesCountValue{ 0 };

			bool brokenEyeEnable{ false };
			double brokenEyeSimilarity{ 0 };

			bool crackEnable{ false };
			double crackSimilarity{ 0 };

			bool apertureEnable{ false };
			double apertureValue{ 0 };
			double apertureSimilarity{ 0 };

			bool holeCenterDistanceEnable{ false };
			double holeCenterDistanceValue{ 0 };
			double holeCenterDistanceSimilarity{ 0 };

			bool specifyColorDifferenceEnable{ false };
			double specifyColorDifferenceR{ 0 };
			double specifyColorDifferenceG{ 0 };
			double specifyColorDifferenceB{ 0 };
			double specifyColorDifferenceDeviation{ 0 };

			bool largeColorDifferenceEnable{ false };
			double largeColorDifferenceDeviation{ 0 };

			bool grindStoneEnable{ false };
			double grindStoneEnableScore{ 0 };//

			bool blockEyeEnable{ false };
			double blockEyeEnableScore{ 0 };//

			bool materialHeadEnable{ false };
			double materialHeadEnableScore{ 0 };//
		};
	}
}