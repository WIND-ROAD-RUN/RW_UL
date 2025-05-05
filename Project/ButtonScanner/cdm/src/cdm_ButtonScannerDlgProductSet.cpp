#include "cdm_ButtonScannerDlgProductSet.h"
#include"oso_core.h"

rw::cdm::ButtonScannerDlgProductSet::ButtonScannerDlgProductSet(const rw::oso::ObjectStoreAssembly& assembly)
{
	auto isAccountAssembly = assembly.getName();
	if (isAccountAssembly != "$class$ButtonScannerDlgProductSet$")
	{
		throw std::runtime_error("Assembly is not $class$ButtonScannerDlgProductSet$");
	}

	auto outsideDiameterEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterEnable$"));
	if (!outsideDiameterEnableItem) {
		throw std::runtime_error("$variable$outsideDiameterEnable is not found");
	}
	outsideDiameterEnable = outsideDiameterEnableItem->getValueAsBool();

	auto outsideDiameterValueItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterValue$"));
	if (!outsideDiameterValueItem) {
		throw std::runtime_error("$variable$outsideDiameterValue is not found");
	}
	outsideDiameterValue = outsideDiameterValueItem->getValueAsDouble();

	auto outsideDiameterDeviationItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterDeviation$"));
	if (!outsideDiameterDeviationItem) {
		throw std::runtime_error("$variable$outsideDiameterDeviation is not found");
	}
	outsideDiameterDeviation = outsideDiameterDeviationItem->getValueAsDouble();

	auto photographyItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$photography$"));
	if (!photographyItem) {
		throw std::runtime_error("$variable$photography is not found");
	}
	photography = photographyItem->getValueAsDouble();

	auto blowTimeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime$"));
	if (!blowTimeItem) {
		throw std::runtime_error("$variable$blowTime is not found");
	}
	blowTime = blowTimeItem->getValueAsDouble();

	auto edgeDamageEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$edgeDamageEnable$"));
	if (!edgeDamageEnableItem) {
		throw std::runtime_error("$variable$edgeDamageEnable is not found");
	}
	edgeDamageEnable = edgeDamageEnableItem->getValueAsBool();

	auto edgeDamageSimilarityItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$edgeDamageSimilarity$"));
	if (!edgeDamageSimilarityItem) {
		throw std::runtime_error("$variable$edgeDamageSimilarity is not found");
	}
	edgeDamageSimilarity = edgeDamageSimilarityItem->getValueAsDouble();

	auto shieldingRangeEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shieldingRangeEnable$"));
	if (!shieldingRangeEnableItem) {
		throw std::runtime_error("$variable$shieldingRangeEnable is not found");
	}
	shieldingRangeEnable = shieldingRangeEnableItem->getValueAsBool();

	auto outerRadiusItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outerRadius$"));
	if (!outerRadiusItem) {
		throw std::runtime_error("$variable$outerRadius is not found");
	}
	outerRadius = outerRadiusItem->getValueAsDouble();

	auto innerRadiusItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$innerRadius$"));
	if (!innerRadiusItem) {
		throw std::runtime_error("$variable$innerRadius is not found");
	}
	innerRadius = innerRadiusItem->getValueAsDouble();

	auto poreEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$poreEnable$"));
	if (!poreEnableItem) {
		throw std::runtime_error("$variable$poreEnable is not found");
	}
	poreEnable = poreEnableItem->getValueAsBool();

	auto paintEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$paintEnable$"));
	if (!paintEnableItem) {
		throw std::runtime_error("$variable$paintEnable is not found");
	}
	paintEnable = paintEnableItem->getValueAsBool();

	auto holesCountEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holesCountEnable$"));
	if (!holesCountEnableItem) {
		throw std::runtime_error("$variable$holesCountEnable is not found");
	}
	holesCountEnable = holesCountEnableItem->getValueAsBool();

	auto holesCountValueItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holesCountValue$"));
	if (!holesCountValueItem) {
		throw std::runtime_error("$variable$holesCountValue is not found");
	}
	holesCountValue = holesCountValueItem->getValueAsDouble();

	auto brokenEyeEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$brokenEyeEnable$"));
	if (!brokenEyeEnableItem) {
		throw std::runtime_error("$variable$brokenEyeEnable is not found");
	}
	brokenEyeEnable = brokenEyeEnableItem->getValueAsBool();

	auto brokenEyeSimilarityItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$brokenEyeSimilarity$"));
	if (!brokenEyeSimilarityItem) {
		throw std::runtime_error("$variable$brokenEyeSimilarity is not found");
	}
	brokenEyeSimilarity = brokenEyeSimilarityItem->getValueAsDouble();

	auto crackSimilarityItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$crackSimilarity$"));
	if (!crackSimilarityItem) {
		throw std::runtime_error("$variable$crackSimilarity is not found");
	}
	crackSimilarity = crackSimilarityItem->getValueAsDouble();

	auto crackEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$crackEnable$"));
	if (!crackEnableItem) {
		throw std::runtime_error("$variable$crackEnable is not found");
	}
	crackEnable = crackEnableItem->getValueAsBool();

	auto apertureEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$apertureEnable$"));
	if (!apertureEnableItem) {
		throw std::runtime_error("$variable$apertureEnable is not found");
	}
	apertureEnable = apertureEnableItem->getValueAsBool();

	auto apertureValueItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$apertureValue$"));
	if (!apertureValueItem) {
		throw std::runtime_error("$variable$apertureValue is not found");
	}
	apertureValue = apertureValueItem->getValueAsDouble();

	auto apertureSimilarityItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$apertureSimilarity$"));
	if (!apertureSimilarityItem) {
		throw std::runtime_error("$variable$apertureSimilarity is not found");
	}
	apertureSimilarity = apertureSimilarityItem->getValueAsDouble();

	auto holeCenterDistanceEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holeCenterDistanceEnable$"));
	if (!holeCenterDistanceEnableItem) {
		throw std::runtime_error("$variable$holeCenterDistanceEnable is not found");
	}
	holeCenterDistanceEnable = holeCenterDistanceEnableItem->getValueAsBool();

	auto holeCenterDistanceValueItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holeCenterDistanceValue$"));
	if (!holeCenterDistanceValueItem) {
		throw std::runtime_error("$variable$holeCenterDistanceValue is not found");
	}
	holeCenterDistanceValue = holeCenterDistanceValueItem->getValueAsDouble();

	auto holeCenterDistanceSimilarityItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holeCenterDistanceSimilarity$"));
	if (!holeCenterDistanceSimilarityItem) {
		throw std::runtime_error("$variable$holeCenterDistanceSimilarity is not found");
	}
	holeCenterDistanceSimilarity = holeCenterDistanceSimilarityItem->getValueAsDouble();

	auto specifyColorDifferenceEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceEnable$"));
	if (!specifyColorDifferenceEnableItem) {
		throw std::runtime_error("$variable$specifyColorDifferenceEnable is not found");
	}
	specifyColorDifferenceEnable = specifyColorDifferenceEnableItem->getValueAsBool();

	auto specifyColorDifferenceRItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceR$"));
	if (!specifyColorDifferenceRItem) {
		throw std::runtime_error("$variable$specifyColorDifferenceR is not found");
	}
	specifyColorDifferenceR = specifyColorDifferenceRItem->getValueAsDouble();

	auto specifyColorDifferenceGItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceG$"));
	if (!specifyColorDifferenceGItem) {
		throw std::runtime_error("$variable$specifyColorDifferenceG is not found");
	}
	specifyColorDifferenceG = specifyColorDifferenceGItem->getValueAsDouble();

	auto specifyColorDifferenceBItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceB$"));
	if (!specifyColorDifferenceBItem) {
		throw std::runtime_error("$variable$specifyColorDifferenceB is not found");
	}
	specifyColorDifferenceB = specifyColorDifferenceBItem->getValueAsDouble();

	auto specifyColorDifferenceDeviationItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceDeviation$"));
	if (!specifyColorDifferenceDeviationItem) {
		throw std::runtime_error("$variable$specifyColorDifferenceDeviation is not found");
	}
	specifyColorDifferenceDeviation = specifyColorDifferenceDeviationItem->getValueAsDouble();

	auto largeColorDifferenceEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$largeColorDifferenceEnable$"));
	if (!largeColorDifferenceEnableItem) {
		throw std::runtime_error("$variable$largeColorDifferenceEnable is not found");
	}
	largeColorDifferenceEnable = largeColorDifferenceEnableItem->getValueAsBool();

	auto largeColorDifferenceDeviationItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$largeColorDifferenceDeviation$"));
	if (!largeColorDifferenceDeviationItem) {
		throw std::runtime_error("$variable$largeColorDifferenceDeviation is not found");
	}
	largeColorDifferenceDeviation = largeColorDifferenceDeviationItem->getValueAsDouble();

	auto grindStoneEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$grindStoneEnable$"));
	if (!grindStoneEnableItem) {
		throw std::runtime_error("$variable$grindStoneEnable is not found");
	}
	grindStoneEnable = grindStoneEnableItem->getValueAsBool();

	auto blockEyeEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blockEyeEnable$"));
	if (!blockEyeEnableItem) {
		throw std::runtime_error("$variable$blockEyeEnable is not found");
	}
	blockEyeEnable = blockEyeEnableItem->getValueAsBool();

	auto materialHeadEnableItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$materialHeadEnable$"));
	if (!materialHeadEnableItem) {
		throw std::runtime_error("$variable$materialHeadEnable is not found");
	}
	materialHeadEnable = materialHeadEnableItem->getValueAsBool();

	auto poreEnableScoreItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$poreEnableScore$"));
	if (!poreEnableScoreItem) {
		throw std::runtime_error("$variable$poreEnableScore is not found");
	}
	poreEnableScore = poreEnableScoreItem->getValueAsDouble();

	auto paintEnableScoreItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$paintEnableScore$"));
	if (!paintEnableScoreItem) {
		throw std::runtime_error("$variable$paintEnableScore is not found");
	}
	paintEnableScore = paintEnableScoreItem->getValueAsDouble();

	auto grindStoneEnableScoreItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$grindStoneEnableScore$"));
	if (!grindStoneEnableScoreItem) {
		throw std::runtime_error("$variable$grindStoneEnableScore is not found");
	}
	grindStoneEnableScore = grindStoneEnableScoreItem->getValueAsDouble();

	auto blockEyeEnableScoreItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blockEyeEnableScore$"));
	if (!blockEyeEnableScoreItem) {
		throw std::runtime_error("$variable$blockEyeEnableScore is not found");
	}
	blockEyeEnableScore = blockEyeEnableScoreItem->getValueAsDouble();

	auto materialHeadEnableScoreItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$materialHeadEnableScore$"));
	if (!materialHeadEnableScoreItem) {
		throw std::runtime_error("$variable$materialHeadEnableScore is not found");
	}
	materialHeadEnableScore = materialHeadEnableScoreItem->getValueAsDouble();
}

rw::cdm::ButtonScannerDlgProductSet::ButtonScannerDlgProductSet(const ButtonScannerDlgProductSet& buttonScannerMainWindow)
{
	outsideDiameterEnable = buttonScannerMainWindow.outsideDiameterEnable;
	outsideDiameterValue = buttonScannerMainWindow.outsideDiameterValue;
	outsideDiameterDeviation = buttonScannerMainWindow.outsideDiameterDeviation;

	photography = buttonScannerMainWindow.photography;
	blowTime = buttonScannerMainWindow.blowTime;

	edgeDamageEnable = buttonScannerMainWindow.edgeDamageEnable;
	edgeDamageSimilarity = buttonScannerMainWindow.edgeDamageSimilarity;

	shieldingRangeEnable = buttonScannerMainWindow.shieldingRangeEnable;
	outerRadius = buttonScannerMainWindow.outerRadius;
	innerRadius = buttonScannerMainWindow.innerRadius;

	poreEnable = buttonScannerMainWindow.poreEnable;
	paintEnable = buttonScannerMainWindow.paintEnable;

	holesCountEnable = buttonScannerMainWindow.holesCountEnable;
	holesCountValue = buttonScannerMainWindow.holesCountValue;

	brokenEyeEnable = buttonScannerMainWindow.brokenEyeEnable;
	brokenEyeSimilarity = buttonScannerMainWindow.brokenEyeSimilarity;

	crackEnable = buttonScannerMainWindow.crackEnable;
	crackSimilarity = buttonScannerMainWindow.crackSimilarity;

	apertureEnable = buttonScannerMainWindow.apertureEnable;
	apertureValue = buttonScannerMainWindow.apertureValue;
	apertureSimilarity = buttonScannerMainWindow.apertureSimilarity;

	holeCenterDistanceEnable = buttonScannerMainWindow.holeCenterDistanceEnable;
	holeCenterDistanceValue = buttonScannerMainWindow.holeCenterDistanceValue;
	holeCenterDistanceSimilarity = buttonScannerMainWindow.holeCenterDistanceSimilarity;

	specifyColorDifferenceEnable = buttonScannerMainWindow.specifyColorDifferenceEnable;
	specifyColorDifferenceR = buttonScannerMainWindow.specifyColorDifferenceR;
	specifyColorDifferenceG = buttonScannerMainWindow.specifyColorDifferenceG;
	specifyColorDifferenceB = buttonScannerMainWindow.specifyColorDifferenceB;
	specifyColorDifferenceDeviation = buttonScannerMainWindow.specifyColorDifferenceDeviation;

	largeColorDifferenceEnable = buttonScannerMainWindow.largeColorDifferenceEnable;
	largeColorDifferenceDeviation = buttonScannerMainWindow.largeColorDifferenceDeviation;

	grindStoneEnable = buttonScannerMainWindow.grindStoneEnable;
	blockEyeEnable = buttonScannerMainWindow.blockEyeEnable;
	materialHeadEnable = buttonScannerMainWindow.materialHeadEnable;

	poreEnableScore = buttonScannerMainWindow.poreEnableScore;
	paintEnableScore = buttonScannerMainWindow.paintEnableScore;
	grindStoneEnableScore = buttonScannerMainWindow.grindStoneEnableScore;
	blockEyeEnableScore = buttonScannerMainWindow.blockEyeEnableScore;
	materialHeadEnableScore = buttonScannerMainWindow.materialHeadEnableScore;
}

rw::cdm::ButtonScannerDlgProductSet& rw::cdm::ButtonScannerDlgProductSet::operator=(const ButtonScannerDlgProductSet& buttonScannerMainWindow)
{
	if (this != &buttonScannerMainWindow) {
		outsideDiameterEnable = buttonScannerMainWindow.outsideDiameterEnable;
		outsideDiameterValue = buttonScannerMainWindow.outsideDiameterValue;
		outsideDiameterDeviation = buttonScannerMainWindow.outsideDiameterDeviation;

		photography = buttonScannerMainWindow.photography;
		blowTime = buttonScannerMainWindow.blowTime;

		edgeDamageEnable = buttonScannerMainWindow.edgeDamageEnable;
		edgeDamageSimilarity = buttonScannerMainWindow.edgeDamageSimilarity;

		shieldingRangeEnable = buttonScannerMainWindow.shieldingRangeEnable;
		outerRadius = buttonScannerMainWindow.outerRadius;
		innerRadius = buttonScannerMainWindow.innerRadius;

		poreEnable = buttonScannerMainWindow.poreEnable;
		paintEnable = buttonScannerMainWindow.paintEnable;

		holesCountEnable = buttonScannerMainWindow.holesCountEnable;
		holesCountValue = buttonScannerMainWindow.holesCountValue;

		brokenEyeEnable = buttonScannerMainWindow.brokenEyeEnable;
		brokenEyeSimilarity = buttonScannerMainWindow.brokenEyeSimilarity;

		crackEnable = buttonScannerMainWindow.crackEnable;
		crackSimilarity = buttonScannerMainWindow.crackSimilarity;

		apertureEnable = buttonScannerMainWindow.apertureEnable;
		apertureValue = buttonScannerMainWindow.apertureValue;
		apertureSimilarity = buttonScannerMainWindow.apertureSimilarity;

		holeCenterDistanceEnable = buttonScannerMainWindow.holeCenterDistanceEnable;
		holeCenterDistanceValue = buttonScannerMainWindow.holeCenterDistanceValue;
		holeCenterDistanceSimilarity = buttonScannerMainWindow.holeCenterDistanceSimilarity;

		specifyColorDifferenceEnable = buttonScannerMainWindow.specifyColorDifferenceEnable;
		specifyColorDifferenceR = buttonScannerMainWindow.specifyColorDifferenceR;
		specifyColorDifferenceG = buttonScannerMainWindow.specifyColorDifferenceG;
		specifyColorDifferenceB = buttonScannerMainWindow.specifyColorDifferenceB;
		specifyColorDifferenceDeviation = buttonScannerMainWindow.specifyColorDifferenceDeviation;

		largeColorDifferenceEnable = buttonScannerMainWindow.largeColorDifferenceEnable;
		largeColorDifferenceDeviation = buttonScannerMainWindow.largeColorDifferenceDeviation;

		grindStoneEnable = buttonScannerMainWindow.grindStoneEnable;
		blockEyeEnable = buttonScannerMainWindow.blockEyeEnable;
		materialHeadEnable = buttonScannerMainWindow.materialHeadEnable;

		poreEnableScore = buttonScannerMainWindow.poreEnableScore;
		paintEnableScore = buttonScannerMainWindow.paintEnableScore;
		grindStoneEnableScore = buttonScannerMainWindow.grindStoneEnableScore;
		blockEyeEnableScore = buttonScannerMainWindow.blockEyeEnableScore;
		materialHeadEnableScore = buttonScannerMainWindow.materialHeadEnableScore;
	}
	return *this;
}

rw::cdm::ButtonScannerDlgProductSet::operator rw::oso::ObjectStoreAssembly() const
{
	rw::oso::ObjectStoreAssembly assembly;
	assembly.setName("$class$ButtonScannerDlgProductSet$");

	auto outsideDiameterEnableItem = std::make_shared<oso::ObjectStoreItem>();
	outsideDiameterEnableItem->setName("$variable$outsideDiameterEnable$");
	outsideDiameterEnableItem->setValueFromBool(outsideDiameterEnable);
	assembly.addItem(outsideDiameterEnableItem);

	auto outsideDiameterValueItem = std::make_shared<oso::ObjectStoreItem>();
	outsideDiameterValueItem->setName("$variable$outsideDiameterValue$");
	outsideDiameterValueItem->setValueFromDouble(outsideDiameterValue);
	assembly.addItem(outsideDiameterValueItem);

	auto outsideDiameterDeviationItem = std::make_shared<oso::ObjectStoreItem>();
	outsideDiameterDeviationItem->setName("$variable$outsideDiameterDeviation$");
	outsideDiameterDeviationItem->setValueFromDouble(outsideDiameterDeviation);
	assembly.addItem(outsideDiameterDeviationItem);

	auto photographyItem = std::make_shared<oso::ObjectStoreItem>();
	photographyItem->setName("$variable$photography$");
	photographyItem->setValueFromDouble(photography);
	assembly.addItem(photographyItem);

	auto blowTimeItem = std::make_shared<oso::ObjectStoreItem>();
	blowTimeItem->setName("$variable$blowTime$");
	blowTimeItem->setValueFromDouble(blowTime);
	assembly.addItem(blowTimeItem);

	auto edgeDamageEnableItem = std::make_shared<oso::ObjectStoreItem>();
	edgeDamageEnableItem->setName("$variable$edgeDamageEnable$");
	edgeDamageEnableItem->setValueFromBool(edgeDamageEnable);
	assembly.addItem(edgeDamageEnableItem);

	auto edgeDamageSimilarityItem = std::make_shared<oso::ObjectStoreItem>();
	edgeDamageSimilarityItem->setName("$variable$edgeDamageSimilarity$");
	edgeDamageSimilarityItem->setValueFromDouble(edgeDamageSimilarity);
	assembly.addItem(edgeDamageSimilarityItem);

	auto shieldingRangeEnableItem = std::make_shared<oso::ObjectStoreItem>();
	shieldingRangeEnableItem->setName("$variable$shieldingRangeEnable$");
	shieldingRangeEnableItem->setValueFromBool(shieldingRangeEnable);
	assembly.addItem(shieldingRangeEnableItem);

	auto outerRadiusItem = std::make_shared<oso::ObjectStoreItem>();
	outerRadiusItem->setName("$variable$outerRadius$");
	outerRadiusItem->setValueFromDouble(outerRadius);
	assembly.addItem(outerRadiusItem);

	auto innerRadiusItem = std::make_shared<oso::ObjectStoreItem>();
	innerRadiusItem->setName("$variable$innerRadius$");
	innerRadiusItem->setValueFromDouble(innerRadius);
	assembly.addItem(innerRadiusItem);

	auto poreEnableItem = std::make_shared<oso::ObjectStoreItem>();
	poreEnableItem->setName("$variable$poreEnable$");
	poreEnableItem->setValueFromBool(poreEnable);
	assembly.addItem(poreEnableItem);

	auto paintEnableItem = std::make_shared<oso::ObjectStoreItem>();
	paintEnableItem->setName("$variable$paintEnable$");
	paintEnableItem->setValueFromBool(paintEnable);
	assembly.addItem(paintEnableItem);

	auto holesCountEnableItem = std::make_shared<oso::ObjectStoreItem>();
	holesCountEnableItem->setName("$variable$holesCountEnable$");
	holesCountEnableItem->setValueFromBool(holesCountEnable);
	assembly.addItem(holesCountEnableItem);

	auto holesCountValueItem = std::make_shared<oso::ObjectStoreItem>();
	holesCountValueItem->setName("$variable$holesCountValue$");
	holesCountValueItem->setValueFromDouble(holesCountValue);
	assembly.addItem(holesCountValueItem);

	auto brokenEyeEnableItem = std::make_shared<oso::ObjectStoreItem>();
	brokenEyeEnableItem->setName("$variable$brokenEyeEnable$");
	brokenEyeEnableItem->setValueFromBool(brokenEyeEnable);
	assembly.addItem(brokenEyeEnableItem);

	auto brokenEyeSimilarityItem = std::make_shared<oso::ObjectStoreItem>();
	brokenEyeSimilarityItem->setName("$variable$brokenEyeSimilarity$");
	brokenEyeSimilarityItem->setValueFromDouble(brokenEyeSimilarity);
	assembly.addItem(brokenEyeSimilarityItem);

	auto crackEnableItem = std::make_shared<oso::ObjectStoreItem>();
	crackEnableItem->setName("$variable$crackEnable$");
	crackEnableItem->setValueFromBool(crackEnable);
	assembly.addItem(crackEnableItem);

	auto crackSimilarityItem = std::make_shared<oso::ObjectStoreItem>();
	crackSimilarityItem->setName("$variable$crackSimilarity$");
	crackSimilarityItem->setValueFromDouble(crackSimilarity);
	assembly.addItem(crackSimilarityItem);

	auto apertureEnableItem = std::make_shared<oso::ObjectStoreItem>();
	apertureEnableItem->setName("$variable$apertureEnable$");
	apertureEnableItem->setValueFromBool(apertureEnable);
	assembly.addItem(apertureEnableItem);

	auto apertureValueItem = std::make_shared<oso::ObjectStoreItem>();
	apertureValueItem->setName("$variable$apertureValue$");
	apertureValueItem->setValueFromDouble(apertureValue);
	assembly.addItem(apertureValueItem);

	auto apertureSimilarityItem = std::make_shared<oso::ObjectStoreItem>();
	apertureSimilarityItem->setName("$variable$apertureSimilarity$");
	apertureSimilarityItem->setValueFromDouble(apertureSimilarity);
	assembly.addItem(apertureSimilarityItem);

	auto holeCenterDistanceEnableItem = std::make_shared<oso::ObjectStoreItem>();
	holeCenterDistanceEnableItem->setName("$variable$holeCenterDistanceEnable$");
	holeCenterDistanceEnableItem->setValueFromBool(holeCenterDistanceEnable);
	assembly.addItem(holeCenterDistanceEnableItem);

	auto holeCenterDistanceValueItem = std::make_shared<oso::ObjectStoreItem>();
	holeCenterDistanceValueItem->setName("$variable$holeCenterDistanceValue$");
	holeCenterDistanceValueItem->setValueFromDouble(holeCenterDistanceValue);
	assembly.addItem(holeCenterDistanceValueItem);

	auto holeCenterDistanceSimilarityItem = std::make_shared<oso::ObjectStoreItem>();
	holeCenterDistanceSimilarityItem->setName("$variable$holeCenterDistanceSimilarity$");
	holeCenterDistanceSimilarityItem->setValueFromDouble(holeCenterDistanceSimilarity);
	assembly.addItem(holeCenterDistanceSimilarityItem);

	auto specifyColorDifferenceEnableItem = std::make_shared<oso::ObjectStoreItem>();
	specifyColorDifferenceEnableItem->setName("$variable$specifyColorDifferenceEnable$");
	specifyColorDifferenceEnableItem->setValueFromBool(specifyColorDifferenceEnable);
	assembly.addItem(specifyColorDifferenceEnableItem);

	auto specifyColorDifferenceRItem = std::make_shared<oso::ObjectStoreItem>();
	specifyColorDifferenceRItem->setName("$variable$specifyColorDifferenceR$");
	specifyColorDifferenceRItem->setValueFromDouble(specifyColorDifferenceR);
	assembly.addItem(specifyColorDifferenceRItem);

	auto specifyColorDifferenceGItem = std::make_shared<oso::ObjectStoreItem>();
	specifyColorDifferenceGItem->setName("$variable$specifyColorDifferenceG$");
	specifyColorDifferenceGItem->setValueFromDouble(specifyColorDifferenceG);
	assembly.addItem(specifyColorDifferenceGItem);

	auto specifyColorDifferenceBItem = std::make_shared<oso::ObjectStoreItem>();
	specifyColorDifferenceBItem->setName("$variable$specifyColorDifferenceB$");
	specifyColorDifferenceBItem->setValueFromDouble(specifyColorDifferenceB);
	assembly.addItem(specifyColorDifferenceBItem);

	auto specifyColorDifferenceDeviationItem = std::make_shared<oso::ObjectStoreItem>();
	specifyColorDifferenceDeviationItem->setName("$variable$specifyColorDifferenceDeviation$");
	specifyColorDifferenceDeviationItem->setValueFromDouble(specifyColorDifferenceDeviation);
	assembly.addItem(specifyColorDifferenceDeviationItem);

	auto largeColorDifferenceEnableItem = std::make_shared<oso::ObjectStoreItem>();
	largeColorDifferenceEnableItem->setName("$variable$largeColorDifferenceEnable$");
	largeColorDifferenceEnableItem->setValueFromBool(largeColorDifferenceEnable);
	assembly.addItem(largeColorDifferenceEnableItem);

	auto largeColorDifferenceDeviationItem = std::make_shared<oso::ObjectStoreItem>();
	largeColorDifferenceDeviationItem->setName("$variable$largeColorDifferenceDeviation$");
	largeColorDifferenceDeviationItem->setValueFromDouble(largeColorDifferenceDeviation);
	assembly.addItem(largeColorDifferenceDeviationItem);

	auto grindStoneEnableItem = std::make_shared<oso::ObjectStoreItem>();
	grindStoneEnableItem->setName("$variable$grindStoneEnable$");
	grindStoneEnableItem->setValueFromBool(grindStoneEnable);
	assembly.addItem(grindStoneEnableItem);

	auto blockEyeEnableItem = std::make_shared<oso::ObjectStoreItem>();
	blockEyeEnableItem->setName("$variable$blockEyeEnable$");
	blockEyeEnableItem->setValueFromBool(blockEyeEnable);
	assembly.addItem(blockEyeEnableItem);

	auto materialHeadEnableItem = std::make_shared<oso::ObjectStoreItem>();
	materialHeadEnableItem->setName("$variable$materialHeadEnable$");
	materialHeadEnableItem->setValueFromBool(materialHeadEnable);
	assembly.addItem(materialHeadEnableItem);

	auto poreEnableScoreItem = std::make_shared<oso::ObjectStoreItem>();
	poreEnableScoreItem->setName("$variable$poreEnableScore$");
	poreEnableScoreItem->setValueFromDouble(poreEnableScore);
	assembly.addItem(poreEnableScoreItem);

	auto paintEnableScoreItem = std::make_shared<oso::ObjectStoreItem>();
	paintEnableScoreItem->setName("$variable$paintEnableScore$");
	paintEnableScoreItem->setValueFromDouble(paintEnableScore);
	assembly.addItem(paintEnableScoreItem);

	auto grindStoneEnableScoreItem = std::make_shared<oso::ObjectStoreItem>();
	grindStoneEnableScoreItem->setName("$variable$grindStoneEnableScore$");
	grindStoneEnableScoreItem->setValueFromDouble(grindStoneEnableScore);
	assembly.addItem(grindStoneEnableScoreItem);

	auto blockEyeEnableScoreItem = std::make_shared<oso::ObjectStoreItem>();
	blockEyeEnableScoreItem->setName("$variable$blockEyeEnableScore$");
	blockEyeEnableScoreItem->setValueFromDouble(blockEyeEnableScore);
	assembly.addItem(blockEyeEnableScoreItem);

	auto materialHeadEnableScoreItem = std::make_shared<oso::ObjectStoreItem>();
	materialHeadEnableScoreItem->setName("$variable$materialHeadEnableScore$");
	materialHeadEnableScoreItem->setValueFromDouble(materialHeadEnableScore);
	assembly.addItem(materialHeadEnableScoreItem);

	return assembly;
}

bool rw::cdm::ButtonScannerDlgProductSet::operator==(const ButtonScannerDlgProductSet& buttonScannerMainWindow) const
{
	return	outsideDiameterEnable == buttonScannerMainWindow.outsideDiameterEnable &&
		outsideDiameterValue == buttonScannerMainWindow.outsideDiameterValue &&
		outsideDiameterDeviation == buttonScannerMainWindow.outsideDiameterDeviation &&

		photography == buttonScannerMainWindow.photography &&
		blowTime == buttonScannerMainWindow.blowTime &&

		edgeDamageEnable == buttonScannerMainWindow.edgeDamageEnable &&
		edgeDamageSimilarity == buttonScannerMainWindow.edgeDamageSimilarity &&

		shieldingRangeEnable == buttonScannerMainWindow.shieldingRangeEnable &&
		outerRadius == buttonScannerMainWindow.outerRadius &&
		innerRadius == buttonScannerMainWindow.innerRadius &&

		poreEnable == buttonScannerMainWindow.poreEnable &&
		paintEnable == buttonScannerMainWindow.paintEnable &&

		holesCountEnable == buttonScannerMainWindow.holesCountEnable &&
		holesCountValue == buttonScannerMainWindow.holesCountValue &&

		brokenEyeEnable == buttonScannerMainWindow.brokenEyeEnable &&
		brokenEyeSimilarity == buttonScannerMainWindow.brokenEyeSimilarity &&

		crackEnable == buttonScannerMainWindow.crackEnable &&
		crackSimilarity == buttonScannerMainWindow.crackSimilarity &&

		apertureEnable == buttonScannerMainWindow.apertureEnable &&
		apertureValue == buttonScannerMainWindow.apertureValue &&
		apertureSimilarity == buttonScannerMainWindow.apertureSimilarity &&

		holeCenterDistanceEnable == buttonScannerMainWindow.holeCenterDistanceEnable &&
		holeCenterDistanceValue == buttonScannerMainWindow.holeCenterDistanceValue &&
		holeCenterDistanceSimilarity == buttonScannerMainWindow.holeCenterDistanceSimilarity &&

		specifyColorDifferenceEnable == buttonScannerMainWindow.specifyColorDifferenceEnable &&
		specifyColorDifferenceR == buttonScannerMainWindow.specifyColorDifferenceR &&
		specifyColorDifferenceG == buttonScannerMainWindow.specifyColorDifferenceG &&
		specifyColorDifferenceB == buttonScannerMainWindow.specifyColorDifferenceB &&
		specifyColorDifferenceDeviation == buttonScannerMainWindow.specifyColorDifferenceDeviation &&

		largeColorDifferenceEnable == buttonScannerMainWindow.largeColorDifferenceEnable &&
		largeColorDifferenceDeviation == buttonScannerMainWindow.largeColorDifferenceDeviation &&

		grindStoneEnable == buttonScannerMainWindow.grindStoneEnable &&
		blockEyeEnable == buttonScannerMainWindow.blockEyeEnable &&
		materialHeadEnable == buttonScannerMainWindow.materialHeadEnable &&

		poreEnableScore == buttonScannerMainWindow.poreEnableScore &&
		paintEnableScore == buttonScannerMainWindow.paintEnableScore &&
		grindStoneEnableScore == buttonScannerMainWindow.grindStoneEnableScore &&
		blockEyeEnableScore == buttonScannerMainWindow.blockEyeEnableScore &&
		materialHeadEnableScore == buttonScannerMainWindow.materialHeadEnableScore;
}

bool rw::cdm::ButtonScannerDlgProductSet::operator!=(const ButtonScannerDlgProductSet& account) const
{
	return !(*this == account);
}