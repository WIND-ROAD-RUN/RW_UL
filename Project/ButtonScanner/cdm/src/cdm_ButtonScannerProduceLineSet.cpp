#include "cdm_ButtonScannerProduceLineSet.h"
#include"oso_core.h"

rw::cdm::ButtonScannerProduceLineSet::ButtonScannerProduceLineSet(const rw::oso::ObjectStoreAssembly& assembly)
{
	auto isAccountAssembly = assembly.getName();
	if (isAccountAssembly != "$class$ButtonScannerProduceLineSet$")
	{
		throw std::runtime_error("Assembly is not $class$ButtonScannerProduceLineSet$");
	}

	auto pulseFactorItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pulseFactor$"));
	if (!pulseFactorItem) {
		throw std::runtime_error("$variable$pulseFactor is not found");
	}
	pulseFactor = pulseFactorItem->getValueAsDouble();

	auto codeWheelItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$codeWheel$"));
	if (!codeWheelItem) {
		throw std::runtime_error("$variable$codeWheel is not found");
	}
	codeWheel = codeWheelItem->getValueAsDouble();

	auto blowingEnable1Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable1$"));
	if (!blowingEnable1Item) {
		throw std::runtime_error("$variable$blowingEnable1 is not found");
	}
	blowingEnable1 = blowingEnable1Item->getValueAsBool();

	auto blowingEnable2Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable2$"));
	if (!blowingEnable2Item) {
		throw std::runtime_error("$variable$blowingEnable2 is not found");
	}
	blowingEnable2 = blowingEnable2Item->getValueAsBool();

	auto blowingEnable3Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable3$"));
	if (!blowingEnable3Item) {
		throw std::runtime_error("$variable$blowingEnable3 is not found");
	}
	blowingEnable3 = blowingEnable3Item->getValueAsBool();

	auto blowingEnable4Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable4$"));
	if (!blowingEnable4Item) {
		throw std::runtime_error("$variable$blowingEnable4 is not found");
	}
	blowingEnable4 = blowingEnable4Item->getValueAsBool();

	auto blowDistance1Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance1$"));
	if (!blowDistance1Item) {
		throw std::runtime_error("$variable$blowDistance1 is not found");
	}
	blowDistance1 = blowDistance1Item->getValueAsDouble();

	auto blowDistance2Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance2$"));
	if (!blowDistance2Item) {
		throw std::runtime_error("$variable$blowDistance2 is not found");
	}
	blowDistance2 = blowDistance2Item->getValueAsDouble();

	auto blowDistance3Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance3$"));
	if (!blowDistance3Item) {
		throw std::runtime_error("$variable$blowDistance3 is not found");
	}
	blowDistance3 = blowDistance3Item->getValueAsDouble();

	auto blowDistance4Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance4$"));
	if (!blowDistance4Item) {
		throw std::runtime_error("$variable$blowDistance4 is not found");
	}
	blowDistance4 = blowDistance4Item->getValueAsDouble();

	auto blowTime1Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime1$"));
	if (!blowTime1Item) {
		throw std::runtime_error("$variable$blowTime1 is not found");
	}
	blowTime1 = blowTime1Item->getValueAsDouble();

	auto blowTime2Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime2$"));
	if (!blowTime2Item) {
		throw std::runtime_error("$variable$blowTime2 is not found");
	}
	blowTime2 = blowTime2Item->getValueAsDouble();

	auto blowTime3Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime3$"));
	if (!blowTime3Item) {
		throw std::runtime_error("$variable$blowTime3 is not found");
	}
	blowTime3 = blowTime3Item->getValueAsDouble();

	auto blowTime4Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime4$"));
	if (!blowTime4Item) {
		throw std::runtime_error("$variable$blowTime4 is not found");
	}
	blowTime4 = blowTime4Item->getValueAsDouble();

	auto pixelEquivalent1Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent1$"));
	if (!pixelEquivalent1Item) {
		throw std::runtime_error("$variable$pixelEquivalent1 is not found");
	}
	pixelEquivalent1 = pixelEquivalent1Item->getValueAsDouble();

	auto pixelEquivalent2Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent2$"));
	if (!pixelEquivalent2Item) {
		throw std::runtime_error("$variable$pixelEquivalent2 is not found");
	}
	pixelEquivalent2 = pixelEquivalent2Item->getValueAsDouble();

	auto pixelEquivalent3Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent3$"));
	if (!pixelEquivalent3Item) {
		throw std::runtime_error("$variable$pixelEquivalent3 is not found");
	}
	pixelEquivalent3 = pixelEquivalent3Item->getValueAsDouble();

	auto pixelEquivalent4Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent4$"));
	if (!pixelEquivalent4Item) {
		throw std::runtime_error("$variable$pixelEquivalent4 is not found");
	}
	pixelEquivalent4 = pixelEquivalent4Item->getValueAsDouble();

	auto limit1Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit1$"));
	if (!limit1Item) {
		throw std::runtime_error("$variable$limit1 is not found");
	}
	limit1 = limit1Item->getValueAsDouble();

	auto limit2Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit2$"));
	if (!limit2Item) {
		throw std::runtime_error("$variable$limit2 is not found");
	}
	limit2 = limit2Item->getValueAsDouble();

	auto limit3Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit3$"));
	if (!limit3Item) {
		throw std::runtime_error("$variable$limit3 is not found");
	}
	limit3 = limit3Item->getValueAsDouble();

	auto limit4Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit4$"));
	if (!limit4Item) {
		throw std::runtime_error("$variable$limit4 is not found");
	}
	limit4 = limit4Item->getValueAsDouble();

	auto minBrightnessItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$minBrightness$"));
	if (!minBrightnessItem) {
		throw std::runtime_error("$variable$minBrightness is not found");
	}
	minBrightness = minBrightnessItem->getValueAsDouble();

	auto maxBrightnessItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$maxBrightness$"));
	if (!maxBrightnessItem) {
		throw std::runtime_error("$variable$maxBrightness is not found");
	}
	maxBrightness = maxBrightnessItem->getValueAsDouble();

	auto powerOnItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$powerOn$"));
	if (!powerOnItem) {
		throw std::runtime_error("$variable$powerOn is not found");
	}
	powerOn = powerOnItem->getValueAsBool();

	auto noneItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$none$"));
	if (!noneItem) {
		throw std::runtime_error("$variable$none is not found");
	}
	none = noneItem->getValueAsBool();

	auto runItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$run$"));
	if (!runItem) {
		throw std::runtime_error("$variable$run is not found");
	}
	run = runItem->getValueAsBool();

	auto alarmItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$alarm$"));
	if (!alarmItem) {
		throw std::runtime_error("$variable$alarm is not found");
	}
	alarm = alarmItem->getValueAsBool();

	auto workstationProtection12Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workstationProtection12$"));
	if (!workstationProtection12Item) {
		throw std::runtime_error("$variable$workstationProtection12 is not found");
	}
	workstationProtection12 = workstationProtection12Item->getValueAsBool();

	auto workstationProtection34Item = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workstationProtection34$"));
	if (!workstationProtection34Item) {
		throw std::runtime_error("$variable$workstationProtection34 is not found");
	}
	workstationProtection34 = workstationProtection34Item->getValueAsBool();

	auto debugModeItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$debugMode$"));
	if (!debugModeItem) {
		throw std::runtime_error("$variable$debugMode is not found");
	}
	debugMode = debugModeItem->getValueAsBool();

	auto motorSpeedItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$motorSpeed$"));
	if (!motorSpeedItem) {
		throw std::runtime_error("$variable$motorSpeed is not found");
	}
	motorSpeed = motorSpeedItem->getValueAsDouble();

	auto beltReductionRatioItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$beltReductionRatio$"));
	if (!beltReductionRatioItem) {
		throw std::runtime_error("$variable$beltReductionRatio is not found");
	}
	beltReductionRatio = beltReductionRatioItem->getValueAsDouble();

	auto accelerationAndDecelerationItem = oso::ObjectStoreCoreToItem(assembly.getItem("$variable$accelerationAndDeceleration$"));
	if (!accelerationAndDecelerationItem) {
		throw std::runtime_error("$variable$accelerationAndDeceleration is not found");
	}
	accelerationAndDeceleration = accelerationAndDecelerationItem->getValueAsDouble();
}

rw::cdm::ButtonScannerProduceLineSet::ButtonScannerProduceLineSet(const ButtonScannerProduceLineSet& buttonScannerMainWindow)
{
	blowingEnable1 = buttonScannerMainWindow.blowingEnable1;
	blowingEnable2 = buttonScannerMainWindow.blowingEnable2;
	blowingEnable3 = buttonScannerMainWindow.blowingEnable3;
	blowingEnable4 = buttonScannerMainWindow.blowingEnable4;

	blowDistance1 = buttonScannerMainWindow.blowDistance1;
	blowDistance2 = buttonScannerMainWindow.blowDistance2;
	blowDistance3 = buttonScannerMainWindow.blowDistance3;
	blowDistance4 = buttonScannerMainWindow.blowDistance4;

	blowTime1 = buttonScannerMainWindow.blowTime1;
	blowTime2 = buttonScannerMainWindow.blowTime2;
	blowTime3 = buttonScannerMainWindow.blowTime3;
	blowTime4 = buttonScannerMainWindow.blowTime4;

	pixelEquivalent1 = buttonScannerMainWindow.pixelEquivalent1;
	pixelEquivalent2 = buttonScannerMainWindow.pixelEquivalent2;
	pixelEquivalent3 = buttonScannerMainWindow.pixelEquivalent3;
	pixelEquivalent4 = buttonScannerMainWindow.pixelEquivalent4;

	limit1 = buttonScannerMainWindow.limit1;
	limit2 = buttonScannerMainWindow.limit2;
	limit3 = buttonScannerMainWindow.limit3;
	limit4 = buttonScannerMainWindow.limit4;

	minBrightness = buttonScannerMainWindow.minBrightness;
	maxBrightness = buttonScannerMainWindow.maxBrightness;

	powerOn = buttonScannerMainWindow.powerOn;
	none = buttonScannerMainWindow.none;
	run = buttonScannerMainWindow.run;
	alarm = buttonScannerMainWindow.alarm;

	workstationProtection12 = buttonScannerMainWindow.workstationProtection12;
	workstationProtection34 = buttonScannerMainWindow.workstationProtection34;
	debugMode = buttonScannerMainWindow.debugMode;

	motorSpeed = buttonScannerMainWindow.motorSpeed;
	beltReductionRatio = buttonScannerMainWindow.beltReductionRatio;
	accelerationAndDeceleration = buttonScannerMainWindow.accelerationAndDeceleration;
	pulseFactor = buttonScannerMainWindow.pulseFactor;
	codeWheel = buttonScannerMainWindow.codeWheel;
}

rw::cdm::ButtonScannerProduceLineSet& rw::cdm::ButtonScannerProduceLineSet::operator=(const ButtonScannerProduceLineSet& buttonScannerMainWindow)
{
	if (this != &buttonScannerMainWindow) {
		blowingEnable1 = buttonScannerMainWindow.blowingEnable1;
		blowingEnable2 = buttonScannerMainWindow.blowingEnable2;
		blowingEnable3 = buttonScannerMainWindow.blowingEnable3;
		blowingEnable4 = buttonScannerMainWindow.blowingEnable4;

		blowDistance1 = buttonScannerMainWindow.blowDistance1;
		blowDistance2 = buttonScannerMainWindow.blowDistance2;
		blowDistance3 = buttonScannerMainWindow.blowDistance3;
		blowDistance4 = buttonScannerMainWindow.blowDistance4;

		blowTime1 = buttonScannerMainWindow.blowTime1;
		blowTime2 = buttonScannerMainWindow.blowTime2;
		blowTime3 = buttonScannerMainWindow.blowTime3;
		blowTime4 = buttonScannerMainWindow.blowTime4;

		pixelEquivalent1 = buttonScannerMainWindow.pixelEquivalent1;
		pixelEquivalent2 = buttonScannerMainWindow.pixelEquivalent2;
		pixelEquivalent3 = buttonScannerMainWindow.pixelEquivalent3;
		pixelEquivalent4 = buttonScannerMainWindow.pixelEquivalent4;

		limit1 = buttonScannerMainWindow.limit1;
		limit2 = buttonScannerMainWindow.limit2;
		limit3 = buttonScannerMainWindow.limit3;
		limit4 = buttonScannerMainWindow.limit4;

		minBrightness = buttonScannerMainWindow.minBrightness;
		maxBrightness = buttonScannerMainWindow.maxBrightness;

		powerOn = buttonScannerMainWindow.powerOn;
		none = buttonScannerMainWindow.none;
		run = buttonScannerMainWindow.run;
		alarm = buttonScannerMainWindow.alarm;

		workstationProtection12 = buttonScannerMainWindow.workstationProtection12;
		workstationProtection34 = buttonScannerMainWindow.workstationProtection34;
		debugMode = buttonScannerMainWindow.debugMode;

		motorSpeed = buttonScannerMainWindow.motorSpeed;
		beltReductionRatio = buttonScannerMainWindow.beltReductionRatio;
		accelerationAndDeceleration = buttonScannerMainWindow.accelerationAndDeceleration;
		pulseFactor = buttonScannerMainWindow.pulseFactor;
		codeWheel = buttonScannerMainWindow.codeWheel;
	}
	return *this;
}

rw::cdm::ButtonScannerProduceLineSet::operator rw::oso::ObjectStoreAssembly() const
{
	rw::oso::ObjectStoreAssembly assembly;
	assembly.setName("$class$ButtonScannerProduceLineSet$");

	auto pulseFactorItem = std::make_shared<oso::ObjectStoreItem>();
	pulseFactorItem->setName("$variable$pulseFactor$");
	pulseFactorItem->setValueFromDouble(pulseFactor);
	assembly.addItem(pulseFactorItem);

	auto codeWheelItem = std::make_shared<oso::ObjectStoreItem>();
	codeWheelItem->setName("$variable$codeWheel$");
	codeWheelItem->setValueFromDouble(codeWheel);
	assembly.addItem(codeWheelItem);

	auto blowingEnable1Item = std::make_shared<oso::ObjectStoreItem>();
	blowingEnable1Item->setName("$variable$blowingEnable1$");
	blowingEnable1Item->setValueFromBool(blowingEnable1);
	assembly.addItem(blowingEnable1Item);

	auto blowingEnable2Item = std::make_shared<oso::ObjectStoreItem>();
	blowingEnable2Item->setName("$variable$blowingEnable2$");
	blowingEnable2Item->setValueFromBool(blowingEnable2);
	assembly.addItem(blowingEnable2Item);

	auto blowingEnable3Item = std::make_shared<oso::ObjectStoreItem>();
	blowingEnable3Item->setName("$variable$blowingEnable3$");
	blowingEnable3Item->setValueFromBool(blowingEnable3);
	assembly.addItem(blowingEnable3Item);

	auto blowingEnable4Item = std::make_shared<oso::ObjectStoreItem>();
	blowingEnable4Item->setName("$variable$blowingEnable4$");
	blowingEnable4Item->setValueFromBool(blowingEnable4);
	assembly.addItem(blowingEnable4Item);

	auto blowDistance1Item = std::make_shared<oso::ObjectStoreItem>();
	blowDistance1Item->setName("$variable$blowDistance1$");
	blowDistance1Item->setValueFromDouble(blowDistance1);
	assembly.addItem(blowDistance1Item);

	auto blowDistance2Item = std::make_shared<oso::ObjectStoreItem>();
	blowDistance2Item->setName("$variable$blowDistance2$");
	blowDistance2Item->setValueFromDouble(blowDistance2);
	assembly.addItem(blowDistance2Item);

	auto blowDistance3Item = std::make_shared<oso::ObjectStoreItem>();
	blowDistance3Item->setName("$variable$blowDistance3$");
	blowDistance3Item->setValueFromDouble(blowDistance3);
	assembly.addItem(blowDistance3Item);

	auto blowDistance4Item = std::make_shared<oso::ObjectStoreItem>();
	blowDistance4Item->setName("$variable$blowDistance4$");
	blowDistance4Item->setValueFromDouble(blowDistance4);
	assembly.addItem(blowDistance4Item);

	auto blowTime1Item = std::make_shared<oso::ObjectStoreItem>();
	blowTime1Item->setName("$variable$blowTime1$");
	blowTime1Item->setValueFromDouble(blowTime1);
	assembly.addItem(blowTime1Item);

	auto blowTime2Item = std::make_shared<oso::ObjectStoreItem>();
	blowTime2Item->setName("$variable$blowTime2$");
	blowTime2Item->setValueFromDouble(blowTime2);
	assembly.addItem(blowTime2Item);

	auto blowTime3Item = std::make_shared<oso::ObjectStoreItem>();
	blowTime3Item->setName("$variable$blowTime3$");
	blowTime3Item->setValueFromDouble(blowTime3);
	assembly.addItem(blowTime3Item);

	auto blowTime4Item = std::make_shared<oso::ObjectStoreItem>();
	blowTime4Item->setName("$variable$blowTime4$");
	blowTime4Item->setValueFromDouble(blowTime4);
	assembly.addItem(blowTime4Item);

	auto pixelEquivalent1Item = std::make_shared<oso::ObjectStoreItem>();
	pixelEquivalent1Item->setName("$variable$pixelEquivalent1$");
	pixelEquivalent1Item->setValueFromDouble(pixelEquivalent1);
	assembly.addItem(pixelEquivalent1Item);

	auto pixelEquivalent2Item = std::make_shared<oso::ObjectStoreItem>();
	pixelEquivalent2Item->setName("$variable$pixelEquivalent2$");
	pixelEquivalent2Item->setValueFromDouble(pixelEquivalent2);
	assembly.addItem(pixelEquivalent2Item);

	auto pixelEquivalent3Item = std::make_shared<oso::ObjectStoreItem>();
	pixelEquivalent3Item->setName("$variable$pixelEquivalent3$");
	pixelEquivalent3Item->setValueFromDouble(pixelEquivalent3);
	assembly.addItem(pixelEquivalent3Item);

	auto pixelEquivalent4Item = std::make_shared<oso::ObjectStoreItem>();
	pixelEquivalent4Item->setName("$variable$pixelEquivalent4$");
	pixelEquivalent4Item->setValueFromDouble(pixelEquivalent4);
	assembly.addItem(pixelEquivalent4Item);

	auto limit1Item = std::make_shared<oso::ObjectStoreItem>();
	limit1Item->setName("$variable$limit1$");
	limit1Item->setValueFromDouble(limit1);
	assembly.addItem(limit1Item);

	auto limit2Item = std::make_shared<oso::ObjectStoreItem>();
	limit2Item->setName("$variable$limit2$");
	limit2Item->setValueFromDouble(limit2);
	assembly.addItem(limit2Item);

	auto limit3Item = std::make_shared<oso::ObjectStoreItem>();
	limit3Item->setName("$variable$limit3$");
	limit3Item->setValueFromDouble(limit3);
	assembly.addItem(limit3Item);

	auto limit4Item = std::make_shared<oso::ObjectStoreItem>();
	limit4Item->setName("$variable$limit4$");
	limit4Item->setValueFromDouble(limit4);
	assembly.addItem(limit4Item);

	auto minBrightnessItem = std::make_shared<oso::ObjectStoreItem>();
	minBrightnessItem->setName("$variable$minBrightness$");
	minBrightnessItem->setValueFromDouble(minBrightness);
	assembly.addItem(minBrightnessItem);

	auto maxBrightnessItem = std::make_shared<oso::ObjectStoreItem>();
	maxBrightnessItem->setName("$variable$maxBrightness$");
	maxBrightnessItem->setValueFromDouble(maxBrightness);
	assembly.addItem(maxBrightnessItem);

	auto powerOnItem = std::make_shared<oso::ObjectStoreItem>();
	powerOnItem->setName("$variable$powerOn$");
	powerOnItem->setValueFromBool(powerOn);
	assembly.addItem(powerOnItem);

	auto noneItem = std::make_shared<oso::ObjectStoreItem>();
	noneItem->setName("$variable$none$");
	noneItem->setValueFromBool(none);
	assembly.addItem(noneItem);

	auto runItem = std::make_shared<oso::ObjectStoreItem>();
	runItem->setName("$variable$run$");
	runItem->setValueFromBool(run);
	assembly.addItem(runItem);

	auto alarmItem = std::make_shared<oso::ObjectStoreItem>();
	alarmItem->setName("$variable$alarm$");
	alarmItem->setValueFromBool(alarm);
	assembly.addItem(alarmItem);

	auto workstationProtection12Item = std::make_shared<oso::ObjectStoreItem>();
	workstationProtection12Item->setName("$variable$workstationProtection12$");
	workstationProtection12Item->setValueFromBool(workstationProtection12);
	assembly.addItem(workstationProtection12Item);

	auto workstationProtection34Item = std::make_shared<oso::ObjectStoreItem>();
	workstationProtection34Item->setName("$variable$workstationProtection34$");
	workstationProtection34Item->setValueFromBool(workstationProtection34);
	assembly.addItem(workstationProtection34Item);

	auto debugModeItem = std::make_shared<oso::ObjectStoreItem>();
	debugModeItem->setName("$variable$debugMode$");
	debugModeItem->setValueFromBool(debugMode);
	assembly.addItem(debugModeItem);

	auto motorSpeedItem = std::make_shared<oso::ObjectStoreItem>();
	motorSpeedItem->setName("$variable$motorSpeed$");
	motorSpeedItem->setValueFromDouble(motorSpeed);
	assembly.addItem(motorSpeedItem);

	auto beltReductionRatioItem = std::make_shared<oso::ObjectStoreItem>();
	beltReductionRatioItem->setName("$variable$beltReductionRatio$");
	beltReductionRatioItem->setValueFromDouble(beltReductionRatio);
	assembly.addItem(beltReductionRatioItem);

	auto accelerationAndDecelerationItem = std::make_shared<oso::ObjectStoreItem>();
	accelerationAndDecelerationItem->setName("$variable$accelerationAndDeceleration$");
	accelerationAndDecelerationItem->setValueFromDouble(accelerationAndDeceleration);
	assembly.addItem(accelerationAndDecelerationItem);

	return assembly;
}

bool rw::cdm::ButtonScannerProduceLineSet::operator==(const ButtonScannerProduceLineSet& buttonScannerMainWindow) const
{
	return	blowingEnable1 == buttonScannerMainWindow.blowingEnable1 &&
		blowingEnable2 == buttonScannerMainWindow.blowingEnable2 &&
		blowingEnable3 == buttonScannerMainWindow.blowingEnable3 &&
		blowingEnable4 == buttonScannerMainWindow.blowingEnable4 &&

		blowDistance1 == buttonScannerMainWindow.blowDistance1 &&
		blowDistance2 == buttonScannerMainWindow.blowDistance2 &&
		blowDistance3 == buttonScannerMainWindow.blowDistance3 &&
		blowDistance4 == buttonScannerMainWindow.blowDistance4 &&

		blowTime1 == buttonScannerMainWindow.blowTime1 &&
		blowTime2 == buttonScannerMainWindow.blowTime2 &&
		blowTime3 == buttonScannerMainWindow.blowTime3 &&
		blowTime4 == buttonScannerMainWindow.blowTime4 &&

		pixelEquivalent1 == buttonScannerMainWindow.pixelEquivalent1 &&
		pixelEquivalent2 == buttonScannerMainWindow.pixelEquivalent2 &&
		pixelEquivalent3 == buttonScannerMainWindow.pixelEquivalent3 &&
		pixelEquivalent4 == buttonScannerMainWindow.pixelEquivalent4 &&

		limit1 == buttonScannerMainWindow.limit1 &&
		limit2 == buttonScannerMainWindow.limit2 &&
		limit3 == buttonScannerMainWindow.limit3 &&
		limit4 == buttonScannerMainWindow.limit4 &&

		minBrightness == buttonScannerMainWindow.minBrightness &&
		maxBrightness == buttonScannerMainWindow.maxBrightness &&

		powerOn == buttonScannerMainWindow.powerOn &&
		none == buttonScannerMainWindow.none &&
		run == buttonScannerMainWindow.run &&
		alarm == buttonScannerMainWindow.alarm &&

		workstationProtection12 == buttonScannerMainWindow.workstationProtection12 &&
		workstationProtection34 == buttonScannerMainWindow.workstationProtection34 &&
		debugMode == buttonScannerMainWindow.debugMode &&

		motorSpeed == buttonScannerMainWindow.motorSpeed &&
		beltReductionRatio == buttonScannerMainWindow.beltReductionRatio &&
		accelerationAndDeceleration == buttonScannerMainWindow.accelerationAndDeceleration &&
		pulseFactor == buttonScannerMainWindow.pulseFactor &&
		codeWheel == buttonScannerMainWindow.codeWheel;
}

bool rw::cdm::ButtonScannerProduceLineSet::operator!=(const ButtonScannerProduceLineSet& account) const
{
	return !(*this == account);
}