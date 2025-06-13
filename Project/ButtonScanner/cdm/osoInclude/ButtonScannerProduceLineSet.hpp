#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ButtonScannerProduceLineSet
    {
    public:
        ButtonScannerProduceLineSet() = default;
        ~ButtonScannerProduceLineSet() = default;

        ButtonScannerProduceLineSet(const rw::oso::ObjectStoreAssembly& assembly);
        ButtonScannerProduceLineSet(const ButtonScannerProduceLineSet& obj);

        ButtonScannerProduceLineSet& operator=(const ButtonScannerProduceLineSet& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ButtonScannerProduceLineSet& obj) const;
        bool operator!=(const ButtonScannerProduceLineSet& obj) const;

    public:
        bool takeNgPictures{ true };
        bool takeMaskPictures{ true };
        bool takeOkPictures{ true };
        bool takePicturesLong{ false };
        bool takeWork1Pictures{ true };
        bool takeWork2Pictures{ true };
        bool takeWork3Pictures{ true };
        bool takeWork4Pictures{ true };
        bool drawRec{ false };
        bool drawCircle{ true };
        bool blowingEnable1{ false };
        bool blowingEnable2{ false };
        bool blowingEnable3{ false };
        bool blowingEnable4{ false };
        double blowDistance1{ 0 };
        double blowDistance2{ 0 };
        double blowDistance3{ 0 };
        double blowDistance4{ 0 };
        double blowTime1{ 0 };
        double blowTime2{ 0 };
        double blowTime3{ 0 };
        double blowTime4{ 0 };
        double pixelEquivalent1{ 0 };
        double pixelEquivalent2{ 0 };
        double pixelEquivalent3{ 0 };
        double pixelEquivalent4{ 0 };
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
    };

    inline ButtonScannerProduceLineSet::ButtonScannerProduceLineSet(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ButtonScannerProduceLineSet$")
        {
            throw std::runtime_error("Assembly is not $class$ButtonScannerProduceLineSet$");
        }
        auto takeNgPicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeNgPictures$"));
        if (!takeNgPicturesItem) {
            throw std::runtime_error("$variable$takeNgPictures is not found");
        }
        takeNgPictures = takeNgPicturesItem->getValueAsBool();
        auto takeMaskPicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeMaskPictures$"));
        if (!takeMaskPicturesItem) {
            throw std::runtime_error("$variable$takeMaskPictures is not found");
        }
        takeMaskPictures = takeMaskPicturesItem->getValueAsBool();
        auto takeOkPicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeOkPictures$"));
        if (!takeOkPicturesItem) {
            throw std::runtime_error("$variable$takeOkPictures is not found");
        }
        takeOkPictures = takeOkPicturesItem->getValueAsBool();
        auto takePicturesLongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takePicturesLong$"));
        if (!takePicturesLongItem) {
            throw std::runtime_error("$variable$takePicturesLong is not found");
        }
        takePicturesLong = takePicturesLongItem->getValueAsBool();
        auto takeWork1PicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeWork1Pictures$"));
        if (!takeWork1PicturesItem) {
            throw std::runtime_error("$variable$takeWork1Pictures is not found");
        }
        takeWork1Pictures = takeWork1PicturesItem->getValueAsBool();
        auto takeWork2PicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeWork2Pictures$"));
        if (!takeWork2PicturesItem) {
            throw std::runtime_error("$variable$takeWork2Pictures is not found");
        }
        takeWork2Pictures = takeWork2PicturesItem->getValueAsBool();
        auto takeWork3PicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeWork3Pictures$"));
        if (!takeWork3PicturesItem) {
            throw std::runtime_error("$variable$takeWork3Pictures is not found");
        }
        takeWork3Pictures = takeWork3PicturesItem->getValueAsBool();
        auto takeWork4PicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeWork4Pictures$"));
        if (!takeWork4PicturesItem) {
            throw std::runtime_error("$variable$takeWork4Pictures is not found");
        }
        takeWork4Pictures = takeWork4PicturesItem->getValueAsBool();
        auto drawRecItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$drawRec$"));
        if (!drawRecItem) {
            throw std::runtime_error("$variable$drawRec is not found");
        }
        drawRec = drawRecItem->getValueAsBool();
        auto drawCircleItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$drawCircle$"));
        if (!drawCircleItem) {
            throw std::runtime_error("$variable$drawCircle is not found");
        }
        drawCircle = drawCircleItem->getValueAsBool();
        auto blowingEnable1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable1$"));
        if (!blowingEnable1Item) {
            throw std::runtime_error("$variable$blowingEnable1 is not found");
        }
        blowingEnable1 = blowingEnable1Item->getValueAsBool();
        auto blowingEnable2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable2$"));
        if (!blowingEnable2Item) {
            throw std::runtime_error("$variable$blowingEnable2 is not found");
        }
        blowingEnable2 = blowingEnable2Item->getValueAsBool();
        auto blowingEnable3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable3$"));
        if (!blowingEnable3Item) {
            throw std::runtime_error("$variable$blowingEnable3 is not found");
        }
        blowingEnable3 = blowingEnable3Item->getValueAsBool();
        auto blowingEnable4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowingEnable4$"));
        if (!blowingEnable4Item) {
            throw std::runtime_error("$variable$blowingEnable4 is not found");
        }
        blowingEnable4 = blowingEnable4Item->getValueAsBool();
        auto blowDistance1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance1$"));
        if (!blowDistance1Item) {
            throw std::runtime_error("$variable$blowDistance1 is not found");
        }
        blowDistance1 = blowDistance1Item->getValueAsDouble();
        auto blowDistance2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance2$"));
        if (!blowDistance2Item) {
            throw std::runtime_error("$variable$blowDistance2 is not found");
        }
        blowDistance2 = blowDistance2Item->getValueAsDouble();
        auto blowDistance3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance3$"));
        if (!blowDistance3Item) {
            throw std::runtime_error("$variable$blowDistance3 is not found");
        }
        blowDistance3 = blowDistance3Item->getValueAsDouble();
        auto blowDistance4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowDistance4$"));
        if (!blowDistance4Item) {
            throw std::runtime_error("$variable$blowDistance4 is not found");
        }
        blowDistance4 = blowDistance4Item->getValueAsDouble();
        auto blowTime1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime1$"));
        if (!blowTime1Item) {
            throw std::runtime_error("$variable$blowTime1 is not found");
        }
        blowTime1 = blowTime1Item->getValueAsDouble();
        auto blowTime2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime2$"));
        if (!blowTime2Item) {
            throw std::runtime_error("$variable$blowTime2 is not found");
        }
        blowTime2 = blowTime2Item->getValueAsDouble();
        auto blowTime3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime3$"));
        if (!blowTime3Item) {
            throw std::runtime_error("$variable$blowTime3 is not found");
        }
        blowTime3 = blowTime3Item->getValueAsDouble();
        auto blowTime4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime4$"));
        if (!blowTime4Item) {
            throw std::runtime_error("$variable$blowTime4 is not found");
        }
        blowTime4 = blowTime4Item->getValueAsDouble();
        auto pixelEquivalent1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent1$"));
        if (!pixelEquivalent1Item) {
            throw std::runtime_error("$variable$pixelEquivalent1 is not found");
        }
        pixelEquivalent1 = pixelEquivalent1Item->getValueAsDouble();
        auto pixelEquivalent2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent2$"));
        if (!pixelEquivalent2Item) {
            throw std::runtime_error("$variable$pixelEquivalent2 is not found");
        }
        pixelEquivalent2 = pixelEquivalent2Item->getValueAsDouble();
        auto pixelEquivalent3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent3$"));
        if (!pixelEquivalent3Item) {
            throw std::runtime_error("$variable$pixelEquivalent3 is not found");
        }
        pixelEquivalent3 = pixelEquivalent3Item->getValueAsDouble();
        auto pixelEquivalent4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pixelEquivalent4$"));
        if (!pixelEquivalent4Item) {
            throw std::runtime_error("$variable$pixelEquivalent4 is not found");
        }
        pixelEquivalent4 = pixelEquivalent4Item->getValueAsDouble();
        auto limit1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit1$"));
        if (!limit1Item) {
            throw std::runtime_error("$variable$limit1 is not found");
        }
        limit1 = limit1Item->getValueAsDouble();
        auto limit2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit2$"));
        if (!limit2Item) {
            throw std::runtime_error("$variable$limit2 is not found");
        }
        limit2 = limit2Item->getValueAsDouble();
        auto limit3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit3$"));
        if (!limit3Item) {
            throw std::runtime_error("$variable$limit3 is not found");
        }
        limit3 = limit3Item->getValueAsDouble();
        auto limit4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$limit4$"));
        if (!limit4Item) {
            throw std::runtime_error("$variable$limit4 is not found");
        }
        limit4 = limit4Item->getValueAsDouble();
        auto minBrightnessItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$minBrightness$"));
        if (!minBrightnessItem) {
            throw std::runtime_error("$variable$minBrightness is not found");
        }
        minBrightness = minBrightnessItem->getValueAsDouble();
        auto maxBrightnessItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$maxBrightness$"));
        if (!maxBrightnessItem) {
            throw std::runtime_error("$variable$maxBrightness is not found");
        }
        maxBrightness = maxBrightnessItem->getValueAsDouble();
        auto powerOnItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$powerOn$"));
        if (!powerOnItem) {
            throw std::runtime_error("$variable$powerOn is not found");
        }
        powerOn = powerOnItem->getValueAsBool();
        auto noneItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$none$"));
        if (!noneItem) {
            throw std::runtime_error("$variable$none is not found");
        }
        none = noneItem->getValueAsBool();
        auto runItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$run$"));
        if (!runItem) {
            throw std::runtime_error("$variable$run is not found");
        }
        run = runItem->getValueAsBool();
        auto alarmItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$alarm$"));
        if (!alarmItem) {
            throw std::runtime_error("$variable$alarm is not found");
        }
        alarm = alarmItem->getValueAsBool();
        auto workstationProtection12Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workstationProtection12$"));
        if (!workstationProtection12Item) {
            throw std::runtime_error("$variable$workstationProtection12 is not found");
        }
        workstationProtection12 = workstationProtection12Item->getValueAsBool();
        auto workstationProtection34Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workstationProtection34$"));
        if (!workstationProtection34Item) {
            throw std::runtime_error("$variable$workstationProtection34 is not found");
        }
        workstationProtection34 = workstationProtection34Item->getValueAsBool();
        auto debugModeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$debugMode$"));
        if (!debugModeItem) {
            throw std::runtime_error("$variable$debugMode is not found");
        }
        debugMode = debugModeItem->getValueAsBool();
        auto motorSpeedItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$motorSpeed$"));
        if (!motorSpeedItem) {
            throw std::runtime_error("$variable$motorSpeed is not found");
        }
        motorSpeed = motorSpeedItem->getValueAsDouble();
        auto beltReductionRatioItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$beltReductionRatio$"));
        if (!beltReductionRatioItem) {
            throw std::runtime_error("$variable$beltReductionRatio is not found");
        }
        beltReductionRatio = beltReductionRatioItem->getValueAsDouble();
        auto accelerationAndDecelerationItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$accelerationAndDeceleration$"));
        if (!accelerationAndDecelerationItem) {
            throw std::runtime_error("$variable$accelerationAndDeceleration is not found");
        }
        accelerationAndDeceleration = accelerationAndDecelerationItem->getValueAsDouble();
        auto codeWheelItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$codeWheel$"));
        if (!codeWheelItem) {
            throw std::runtime_error("$variable$codeWheel is not found");
        }
        codeWheel = codeWheelItem->getValueAsDouble();
        auto pulseFactorItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pulseFactor$"));
        if (!pulseFactorItem) {
            throw std::runtime_error("$variable$pulseFactor is not found");
        }
        pulseFactor = pulseFactorItem->getValueAsDouble();
    }

    inline ButtonScannerProduceLineSet::ButtonScannerProduceLineSet(const ButtonScannerProduceLineSet& obj)
    {
        takeNgPictures = obj.takeNgPictures;
        takeMaskPictures = obj.takeMaskPictures;
        takeOkPictures = obj.takeOkPictures;
        takePicturesLong = obj.takePicturesLong;
        takeWork1Pictures = obj.takeWork1Pictures;
        takeWork2Pictures = obj.takeWork2Pictures;
        takeWork3Pictures = obj.takeWork3Pictures;
        takeWork4Pictures = obj.takeWork4Pictures;
        drawRec = obj.drawRec;
        drawCircle = obj.drawCircle;
        blowingEnable1 = obj.blowingEnable1;
        blowingEnable2 = obj.blowingEnable2;
        blowingEnable3 = obj.blowingEnable3;
        blowingEnable4 = obj.blowingEnable4;
        blowDistance1 = obj.blowDistance1;
        blowDistance2 = obj.blowDistance2;
        blowDistance3 = obj.blowDistance3;
        blowDistance4 = obj.blowDistance4;
        blowTime1 = obj.blowTime1;
        blowTime2 = obj.blowTime2;
        blowTime3 = obj.blowTime3;
        blowTime4 = obj.blowTime4;
        pixelEquivalent1 = obj.pixelEquivalent1;
        pixelEquivalent2 = obj.pixelEquivalent2;
        pixelEquivalent3 = obj.pixelEquivalent3;
        pixelEquivalent4 = obj.pixelEquivalent4;
        limit1 = obj.limit1;
        limit2 = obj.limit2;
        limit3 = obj.limit3;
        limit4 = obj.limit4;
        minBrightness = obj.minBrightness;
        maxBrightness = obj.maxBrightness;
        powerOn = obj.powerOn;
        none = obj.none;
        run = obj.run;
        alarm = obj.alarm;
        workstationProtection12 = obj.workstationProtection12;
        workstationProtection34 = obj.workstationProtection34;
        debugMode = obj.debugMode;
        motorSpeed = obj.motorSpeed;
        beltReductionRatio = obj.beltReductionRatio;
        accelerationAndDeceleration = obj.accelerationAndDeceleration;
        codeWheel = obj.codeWheel;
        pulseFactor = obj.pulseFactor;
    }

    inline ButtonScannerProduceLineSet& ButtonScannerProduceLineSet::operator=(const ButtonScannerProduceLineSet& obj)
    {
        if (this != &obj) {
            takeNgPictures = obj.takeNgPictures;
            takeMaskPictures = obj.takeMaskPictures;
            takeOkPictures = obj.takeOkPictures;
            takePicturesLong = obj.takePicturesLong;
            takeWork1Pictures = obj.takeWork1Pictures;
            takeWork2Pictures = obj.takeWork2Pictures;
            takeWork3Pictures = obj.takeWork3Pictures;
            takeWork4Pictures = obj.takeWork4Pictures;
            drawRec = obj.drawRec;
            drawCircle = obj.drawCircle;
            blowingEnable1 = obj.blowingEnable1;
            blowingEnable2 = obj.blowingEnable2;
            blowingEnable3 = obj.blowingEnable3;
            blowingEnable4 = obj.blowingEnable4;
            blowDistance1 = obj.blowDistance1;
            blowDistance2 = obj.blowDistance2;
            blowDistance3 = obj.blowDistance3;
            blowDistance4 = obj.blowDistance4;
            blowTime1 = obj.blowTime1;
            blowTime2 = obj.blowTime2;
            blowTime3 = obj.blowTime3;
            blowTime4 = obj.blowTime4;
            pixelEquivalent1 = obj.pixelEquivalent1;
            pixelEquivalent2 = obj.pixelEquivalent2;
            pixelEquivalent3 = obj.pixelEquivalent3;
            pixelEquivalent4 = obj.pixelEquivalent4;
            limit1 = obj.limit1;
            limit2 = obj.limit2;
            limit3 = obj.limit3;
            limit4 = obj.limit4;
            minBrightness = obj.minBrightness;
            maxBrightness = obj.maxBrightness;
            powerOn = obj.powerOn;
            none = obj.none;
            run = obj.run;
            alarm = obj.alarm;
            workstationProtection12 = obj.workstationProtection12;
            workstationProtection34 = obj.workstationProtection34;
            debugMode = obj.debugMode;
            motorSpeed = obj.motorSpeed;
            beltReductionRatio = obj.beltReductionRatio;
            accelerationAndDeceleration = obj.accelerationAndDeceleration;
            codeWheel = obj.codeWheel;
            pulseFactor = obj.pulseFactor;
        }
        return *this;
    }

    inline ButtonScannerProduceLineSet::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ButtonScannerProduceLineSet$");
        auto takeNgPicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeNgPicturesItem->setName("$variable$takeNgPictures$");
        takeNgPicturesItem->setValueFromBool(takeNgPictures);
        assembly.addItem(takeNgPicturesItem);
        auto takeMaskPicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeMaskPicturesItem->setName("$variable$takeMaskPictures$");
        takeMaskPicturesItem->setValueFromBool(takeMaskPictures);
        assembly.addItem(takeMaskPicturesItem);
        auto takeOkPicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeOkPicturesItem->setName("$variable$takeOkPictures$");
        takeOkPicturesItem->setValueFromBool(takeOkPictures);
        assembly.addItem(takeOkPicturesItem);
        auto takePicturesLongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takePicturesLongItem->setName("$variable$takePicturesLong$");
        takePicturesLongItem->setValueFromBool(takePicturesLong);
        assembly.addItem(takePicturesLongItem);
        auto takeWork1PicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeWork1PicturesItem->setName("$variable$takeWork1Pictures$");
        takeWork1PicturesItem->setValueFromBool(takeWork1Pictures);
        assembly.addItem(takeWork1PicturesItem);
        auto takeWork2PicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeWork2PicturesItem->setName("$variable$takeWork2Pictures$");
        takeWork2PicturesItem->setValueFromBool(takeWork2Pictures);
        assembly.addItem(takeWork2PicturesItem);
        auto takeWork3PicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeWork3PicturesItem->setName("$variable$takeWork3Pictures$");
        takeWork3PicturesItem->setValueFromBool(takeWork3Pictures);
        assembly.addItem(takeWork3PicturesItem);
        auto takeWork4PicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeWork4PicturesItem->setName("$variable$takeWork4Pictures$");
        takeWork4PicturesItem->setValueFromBool(takeWork4Pictures);
        assembly.addItem(takeWork4PicturesItem);
        auto drawRecItem = std::make_shared<rw::oso::ObjectStoreItem>();
        drawRecItem->setName("$variable$drawRec$");
        drawRecItem->setValueFromBool(drawRec);
        assembly.addItem(drawRecItem);
        auto drawCircleItem = std::make_shared<rw::oso::ObjectStoreItem>();
        drawCircleItem->setName("$variable$drawCircle$");
        drawCircleItem->setValueFromBool(drawCircle);
        assembly.addItem(drawCircleItem);
        auto blowingEnable1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowingEnable1Item->setName("$variable$blowingEnable1$");
        blowingEnable1Item->setValueFromBool(blowingEnable1);
        assembly.addItem(blowingEnable1Item);
        auto blowingEnable2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowingEnable2Item->setName("$variable$blowingEnable2$");
        blowingEnable2Item->setValueFromBool(blowingEnable2);
        assembly.addItem(blowingEnable2Item);
        auto blowingEnable3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowingEnable3Item->setName("$variable$blowingEnable3$");
        blowingEnable3Item->setValueFromBool(blowingEnable3);
        assembly.addItem(blowingEnable3Item);
        auto blowingEnable4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowingEnable4Item->setName("$variable$blowingEnable4$");
        blowingEnable4Item->setValueFromBool(blowingEnable4);
        assembly.addItem(blowingEnable4Item);
        auto blowDistance1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowDistance1Item->setName("$variable$blowDistance1$");
        blowDistance1Item->setValueFromDouble(blowDistance1);
        assembly.addItem(blowDistance1Item);
        auto blowDistance2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowDistance2Item->setName("$variable$blowDistance2$");
        blowDistance2Item->setValueFromDouble(blowDistance2);
        assembly.addItem(blowDistance2Item);
        auto blowDistance3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowDistance3Item->setName("$variable$blowDistance3$");
        blowDistance3Item->setValueFromDouble(blowDistance3);
        assembly.addItem(blowDistance3Item);
        auto blowDistance4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowDistance4Item->setName("$variable$blowDistance4$");
        blowDistance4Item->setValueFromDouble(blowDistance4);
        assembly.addItem(blowDistance4Item);
        auto blowTime1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowTime1Item->setName("$variable$blowTime1$");
        blowTime1Item->setValueFromDouble(blowTime1);
        assembly.addItem(blowTime1Item);
        auto blowTime2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowTime2Item->setName("$variable$blowTime2$");
        blowTime2Item->setValueFromDouble(blowTime2);
        assembly.addItem(blowTime2Item);
        auto blowTime3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowTime3Item->setName("$variable$blowTime3$");
        blowTime3Item->setValueFromDouble(blowTime3);
        assembly.addItem(blowTime3Item);
        auto blowTime4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        blowTime4Item->setName("$variable$blowTime4$");
        blowTime4Item->setValueFromDouble(blowTime4);
        assembly.addItem(blowTime4Item);
        auto pixelEquivalent1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        pixelEquivalent1Item->setName("$variable$pixelEquivalent1$");
        pixelEquivalent1Item->setValueFromDouble(pixelEquivalent1);
        assembly.addItem(pixelEquivalent1Item);
        auto pixelEquivalent2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        pixelEquivalent2Item->setName("$variable$pixelEquivalent2$");
        pixelEquivalent2Item->setValueFromDouble(pixelEquivalent2);
        assembly.addItem(pixelEquivalent2Item);
        auto pixelEquivalent3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        pixelEquivalent3Item->setName("$variable$pixelEquivalent3$");
        pixelEquivalent3Item->setValueFromDouble(pixelEquivalent3);
        assembly.addItem(pixelEquivalent3Item);
        auto pixelEquivalent4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        pixelEquivalent4Item->setName("$variable$pixelEquivalent4$");
        pixelEquivalent4Item->setValueFromDouble(pixelEquivalent4);
        assembly.addItem(pixelEquivalent4Item);
        auto limit1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        limit1Item->setName("$variable$limit1$");
        limit1Item->setValueFromDouble(limit1);
        assembly.addItem(limit1Item);
        auto limit2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        limit2Item->setName("$variable$limit2$");
        limit2Item->setValueFromDouble(limit2);
        assembly.addItem(limit2Item);
        auto limit3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        limit3Item->setName("$variable$limit3$");
        limit3Item->setValueFromDouble(limit3);
        assembly.addItem(limit3Item);
        auto limit4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        limit4Item->setName("$variable$limit4$");
        limit4Item->setValueFromDouble(limit4);
        assembly.addItem(limit4Item);
        auto minBrightnessItem = std::make_shared<rw::oso::ObjectStoreItem>();
        minBrightnessItem->setName("$variable$minBrightness$");
        minBrightnessItem->setValueFromDouble(minBrightness);
        assembly.addItem(minBrightnessItem);
        auto maxBrightnessItem = std::make_shared<rw::oso::ObjectStoreItem>();
        maxBrightnessItem->setName("$variable$maxBrightness$");
        maxBrightnessItem->setValueFromDouble(maxBrightness);
        assembly.addItem(maxBrightnessItem);
        auto powerOnItem = std::make_shared<rw::oso::ObjectStoreItem>();
        powerOnItem->setName("$variable$powerOn$");
        powerOnItem->setValueFromBool(powerOn);
        assembly.addItem(powerOnItem);
        auto noneItem = std::make_shared<rw::oso::ObjectStoreItem>();
        noneItem->setName("$variable$none$");
        noneItem->setValueFromBool(none);
        assembly.addItem(noneItem);
        auto runItem = std::make_shared<rw::oso::ObjectStoreItem>();
        runItem->setName("$variable$run$");
        runItem->setValueFromBool(run);
        assembly.addItem(runItem);
        auto alarmItem = std::make_shared<rw::oso::ObjectStoreItem>();
        alarmItem->setName("$variable$alarm$");
        alarmItem->setValueFromBool(alarm);
        assembly.addItem(alarmItem);
        auto workstationProtection12Item = std::make_shared<rw::oso::ObjectStoreItem>();
        workstationProtection12Item->setName("$variable$workstationProtection12$");
        workstationProtection12Item->setValueFromBool(workstationProtection12);
        assembly.addItem(workstationProtection12Item);
        auto workstationProtection34Item = std::make_shared<rw::oso::ObjectStoreItem>();
        workstationProtection34Item->setName("$variable$workstationProtection34$");
        workstationProtection34Item->setValueFromBool(workstationProtection34);
        assembly.addItem(workstationProtection34Item);
        auto debugModeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        debugModeItem->setName("$variable$debugMode$");
        debugModeItem->setValueFromBool(debugMode);
        assembly.addItem(debugModeItem);
        auto motorSpeedItem = std::make_shared<rw::oso::ObjectStoreItem>();
        motorSpeedItem->setName("$variable$motorSpeed$");
        motorSpeedItem->setValueFromDouble(motorSpeed);
        assembly.addItem(motorSpeedItem);
        auto beltReductionRatioItem = std::make_shared<rw::oso::ObjectStoreItem>();
        beltReductionRatioItem->setName("$variable$beltReductionRatio$");
        beltReductionRatioItem->setValueFromDouble(beltReductionRatio);
        assembly.addItem(beltReductionRatioItem);
        auto accelerationAndDecelerationItem = std::make_shared<rw::oso::ObjectStoreItem>();
        accelerationAndDecelerationItem->setName("$variable$accelerationAndDeceleration$");
        accelerationAndDecelerationItem->setValueFromDouble(accelerationAndDeceleration);
        assembly.addItem(accelerationAndDecelerationItem);
        auto codeWheelItem = std::make_shared<rw::oso::ObjectStoreItem>();
        codeWheelItem->setName("$variable$codeWheel$");
        codeWheelItem->setValueFromDouble(codeWheel);
        assembly.addItem(codeWheelItem);
        auto pulseFactorItem = std::make_shared<rw::oso::ObjectStoreItem>();
        pulseFactorItem->setName("$variable$pulseFactor$");
        pulseFactorItem->setValueFromDouble(pulseFactor);
        assembly.addItem(pulseFactorItem);
        return assembly;
    }

    inline bool ButtonScannerProduceLineSet::operator==(const ButtonScannerProduceLineSet& obj) const
    {
        return takeNgPictures == obj.takeNgPictures && takeMaskPictures == obj.takeMaskPictures && takeOkPictures == obj.takeOkPictures && takePicturesLong == obj.takePicturesLong && takeWork1Pictures == obj.takeWork1Pictures && takeWork2Pictures == obj.takeWork2Pictures && takeWork3Pictures == obj.takeWork3Pictures && takeWork4Pictures == obj.takeWork4Pictures && drawRec == obj.drawRec && drawCircle == obj.drawCircle && blowingEnable1 == obj.blowingEnable1 && blowingEnable2 == obj.blowingEnable2 && blowingEnable3 == obj.blowingEnable3 && blowingEnable4 == obj.blowingEnable4 && blowDistance1 == obj.blowDistance1 && blowDistance2 == obj.blowDistance2 && blowDistance3 == obj.blowDistance3 && blowDistance4 == obj.blowDistance4 && blowTime1 == obj.blowTime1 && blowTime2 == obj.blowTime2 && blowTime3 == obj.blowTime3 && blowTime4 == obj.blowTime4 && pixelEquivalent1 == obj.pixelEquivalent1 && pixelEquivalent2 == obj.pixelEquivalent2 && pixelEquivalent3 == obj.pixelEquivalent3 && pixelEquivalent4 == obj.pixelEquivalent4 && limit1 == obj.limit1 && limit2 == obj.limit2 && limit3 == obj.limit3 && limit4 == obj.limit4 && minBrightness == obj.minBrightness && maxBrightness == obj.maxBrightness && powerOn == obj.powerOn && none == obj.none && run == obj.run && alarm == obj.alarm && workstationProtection12 == obj.workstationProtection12 && workstationProtection34 == obj.workstationProtection34 && debugMode == obj.debugMode && motorSpeed == obj.motorSpeed && beltReductionRatio == obj.beltReductionRatio && accelerationAndDeceleration == obj.accelerationAndDeceleration && codeWheel == obj.codeWheel && pulseFactor == obj.pulseFactor;
    }

    inline bool ButtonScannerProduceLineSet::operator!=(const ButtonScannerProduceLineSet& obj) const
    {
        return !(*this == obj);
    }

}

