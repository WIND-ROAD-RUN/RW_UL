#include"cdm_ButtonScannerMainWindow.h"

#include"oso_core.h"

rw::cdm::ButtonScannerMainWindow::ButtonScannerMainWindow(const rw::oso::ObjectStoreAssembly& assembly)
{
	auto isAccountAssembly = assembly.getName();
	if (isAccountAssembly != "$class$ButtonScannerMainWindow$")
	{
		throw std::runtime_error("Assembly is not $class$ButtonScannerMainWindow$");
	}

	auto totalProductionItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$totalProduction$"));
	if (!totalProductionItem) {
		throw std::runtime_error("$variable$totalProduction is not found");
	}
	totalProduction = totalProductionItem->getValueAsLong();

	auto totalWasteItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$totalWaste$"));
	if (!totalWasteItem) {
		throw std::runtime_error("$variable$totalWaste is not found");
	}
	totalWaste = totalWasteItem->getValueAsLong();

	auto passRateItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$passRate$"));
	if (!passRateItem) {
		throw std::runtime_error("$variable$passRate is not found");
	}
	passRate = passRateItem->getValueAsDouble();

	auto isDebugModeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isDebugMode$"));
	if (!isDebugModeItem) {
		throw std::runtime_error("$variable$isDebugMode is not found");
	}
	isDebugMode = isDebugModeItem->getValueAsBool();

	auto isTakePicturesItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isTakePictures$"));
	if (!isTakePicturesItem) {
		throw std::runtime_error("$variable$isTakePictures is not found");
	}
	isTakePictures = isTakePicturesItem->getValueAsBool();

	auto isEliminatingItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isEliminating$"));
	if (!isEliminatingItem) {
		throw std::runtime_error("$variable$isEliminating is not found");
	}
	isEliminating = isEliminatingItem->getValueAsBool();

	auto scrappingRateItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$scrappingRate$"));
	if (!scrappingRateItem) {
		throw std::runtime_error("$variable$scrappingRate is not found");
	}
	scrappingRate = scrappingRateItem->getValueAsBool();

	auto upLightItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$upLight$"));
	if (!upLightItem) {
		throw std::runtime_error("$variable$upLight is not found");
	}
	upLight = upLightItem->getValueAsBool();

	auto downLightItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$downLight$"));
	if (!downLightItem) {
		throw std::runtime_error("$variable$downLight is not found");
	}
	downLight = downLightItem->getValueAsBool();

	auto sideLightItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$sideLight$"));
	if (!sideLightItem) {
		throw std::runtime_error("$variable$sideLight is not found");
	}
	sideLight = sideLightItem->getValueAsBool();

	auto lightValueItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$beltSpeed$"));
	if (!lightValueItem) {
		throw std::runtime_error("$variable$beltSpeed$ is not found");
	}
	beltSpeed = lightValueItem->getValueAsDouble();

	auto speedItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$speed$"));
	if (!speedItem) {
		throw std::runtime_error("$variable$speed is not found");
	}
	speed = speedItem->getValueAsDouble();

	auto isDefectItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isDefect$"));
	if (!isDefectItem) {
		throw std::runtime_error("$variable$isDefect is not found");
	}
	isDefect = isDefectItem->getValueAsBool();

	auto isPositiveItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isPositive$"));
	if (!isPositiveItem) {
		throw std::runtime_error("$variable$isPositive is not found");
	}
	isPositive = isPositiveItem->getValueAsBool();
}

rw::cdm::ButtonScannerMainWindow::ButtonScannerMainWindow(const ButtonScannerMainWindow& buttonScannerMainWindow)
{
	totalProduction = buttonScannerMainWindow.totalProduction;
	totalWaste = buttonScannerMainWindow.totalWaste;
	passRate = buttonScannerMainWindow.passRate;
	isDebugMode = buttonScannerMainWindow.isDebugMode;
	isTakePictures = buttonScannerMainWindow.isTakePictures;
	beltSpeed = buttonScannerMainWindow.beltSpeed;
	isEliminating = buttonScannerMainWindow.isEliminating;
	scrappingRate = buttonScannerMainWindow.scrappingRate;
	upLight = buttonScannerMainWindow.upLight;
	downLight = buttonScannerMainWindow.downLight;
	sideLight = buttonScannerMainWindow.sideLight;
	speed = buttonScannerMainWindow.speed;
	isDefect = buttonScannerMainWindow.isDefect;
	isPositive = buttonScannerMainWindow.isPositive;
}

rw::cdm::ButtonScannerMainWindow& rw::cdm::ButtonScannerMainWindow::operator=(const ButtonScannerMainWindow& buttonScannerMainWindow)
{
	if (this != &buttonScannerMainWindow) {
		totalProduction = buttonScannerMainWindow.totalProduction;
		totalWaste = buttonScannerMainWindow.totalWaste;
		passRate = buttonScannerMainWindow.passRate;
		beltSpeed = buttonScannerMainWindow.beltSpeed;
		isDebugMode = buttonScannerMainWindow.isDebugMode;
		isTakePictures = buttonScannerMainWindow.isTakePictures;
		isEliminating = buttonScannerMainWindow.isEliminating;
		scrappingRate = buttonScannerMainWindow.scrappingRate;
		upLight = buttonScannerMainWindow.upLight;
		downLight = buttonScannerMainWindow.downLight;
		sideLight = buttonScannerMainWindow.sideLight;
		speed = buttonScannerMainWindow.speed;
		isDefect = buttonScannerMainWindow.isDefect;
		isPositive = buttonScannerMainWindow.isPositive;
	}
	return *this;
}

rw::cdm::ButtonScannerMainWindow::operator rw::oso::ObjectStoreAssembly() const
{
	rw::oso::ObjectStoreAssembly assembly;
	assembly.setName("$class$ButtonScannerMainWindow$");

	auto totalProductionItem = std::make_shared<oso::ObjectStoreItem>();
	totalProductionItem->setName("$variable$totalProduction$");
	totalProductionItem->setValueFromLong(totalProduction);
	assembly.addItem(totalProductionItem);

	auto totalWasteItem = std::make_shared<oso::ObjectStoreItem>();
	totalWasteItem->setName("$variable$totalWaste$");
	totalWasteItem->setValueFromLong(totalWaste);
	assembly.addItem(totalWasteItem);

	auto passRateItem = std::make_shared<oso::ObjectStoreItem>();
	passRateItem->setName("$variable$passRate$");
	passRateItem->setValueFromDouble(passRate);
	assembly.addItem(passRateItem);

	auto isDebugModeItem = std::make_shared<oso::ObjectStoreItem>();
	isDebugModeItem->setName("$variable$isDebugMode$");
	isDebugModeItem->setValueFromBool(isDebugMode);
	assembly.addItem(isDebugModeItem);

	auto isTakePicturesItem = std::make_shared<oso::ObjectStoreItem>();
	isTakePicturesItem->setName("$variable$isTakePictures$");
	isTakePicturesItem->setValueFromBool(isTakePictures);
	assembly.addItem(isTakePicturesItem);

	auto isEliminatingItem = std::make_shared<oso::ObjectStoreItem>();
	isEliminatingItem->setName("$variable$isEliminating$");
	isEliminatingItem->setValueFromBool(isEliminating);
	assembly.addItem(isEliminatingItem);

	auto scrappingRateItem = std::make_shared<oso::ObjectStoreItem>();
	scrappingRateItem->setName("$variable$scrappingRate$");
	scrappingRateItem->setValueFromBool(scrappingRate);
	assembly.addItem(scrappingRateItem);

	auto upLightItem = std::make_shared<oso::ObjectStoreItem>();
	upLightItem->setName("$variable$upLight$");
	upLightItem->setValueFromBool(upLight);
	assembly.addItem(upLightItem);

	auto lightValueItem = std::make_shared<oso::ObjectStoreItem>();
	lightValueItem->setName("$variable$beltSpeed$");
	lightValueItem->setValueFromDouble(beltSpeed);
	assembly.addItem(lightValueItem);

	auto downLightItem = std::make_shared<oso::ObjectStoreItem>();
	downLightItem->setName("$variable$downLight$");
	downLightItem->setValueFromBool(downLight);
	assembly.addItem(downLightItem);

	auto sideLightItem = std::make_shared<oso::ObjectStoreItem>();
	sideLightItem->setName("$variable$sideLight$");
	sideLightItem->setValueFromBool(sideLight);
	assembly.addItem(sideLightItem);

	auto speedItem = std::make_shared<oso::ObjectStoreItem>();
	speedItem->setName("$variable$speed$");
	speedItem->setValueFromDouble(speed);
	assembly.addItem(speedItem);

	auto isDefectItem = std::make_shared<oso::ObjectStoreItem>();
	isDefectItem->setName("$variable$isDefect$");
	isDefectItem->setValueFromBool(isDefect);
	assembly.addItem(isDefectItem);

	auto isPositiveItem = std::make_shared<oso::ObjectStoreItem>();
	isPositiveItem->setName("$variable$isPositive$");
	isPositiveItem->setValueFromBool(isPositive);
	assembly.addItem(isPositiveItem);

	return assembly;
}

bool rw::cdm::ButtonScannerMainWindow::operator==(const ButtonScannerMainWindow& account) const
{
	return totalProduction == account.totalProduction &&
		totalWaste == account.totalWaste &&
		passRate == account.passRate &&
		beltSpeed == account.beltSpeed &&
		isDebugMode == account.isDebugMode &&
		isTakePictures == account.isTakePictures &&
		isEliminating == account.isEliminating &&
		scrappingRate == account.scrappingRate &&
		upLight == account.upLight &&
		downLight == account.downLight &&
		sideLight == account.sideLight &&
		speed == account.speed &&
		isDefect == account.isDefect &&
		isPositive == account.isPositive;
}

bool rw::cdm::ButtonScannerMainWindow::operator!=(const ButtonScannerMainWindow& account) const
{
	return !(*this == account);
}