#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ButtonScannerMainWindow
    {
    public:
        ButtonScannerMainWindow() = default;
        ~ButtonScannerMainWindow() = default;

        ButtonScannerMainWindow(const rw::oso::ObjectStoreAssembly& assembly);
        ButtonScannerMainWindow(const ButtonScannerMainWindow& obj);

        ButtonScannerMainWindow& operator=(const ButtonScannerMainWindow& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ButtonScannerMainWindow& obj) const;
        bool operator!=(const ButtonScannerMainWindow& obj) const;

    public:
        long totalProduction{ 0 };
        long totalWaste{ 0 };
        double passRate{ 0 };
        bool isDebugMode{ false };
        bool isTakePictures{ false };
        bool isEliminating{ false };
        bool scrappingRate{ false };
        bool upLight{ false };
        bool downLight{ false };
        bool sideLight{ false };
        bool strobeLight{ false };
        double speed{ 0 };
        double beltSpeed{ 0 };
        bool isDefect{ false };
        bool isPositive{ false };
    };

    inline ButtonScannerMainWindow::ButtonScannerMainWindow(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ButtonScannerMainWindow$")
        {
            throw std::runtime_error("Assembly is not $class$ButtonScannerMainWindow$");
        }
        auto totalProductionItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$totalProduction$"));
        if (!totalProductionItem) {
            throw std::runtime_error("$variable$totalProduction is not found");
        }
        totalProduction = totalProductionItem->getValueAsLong();
        auto totalWasteItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$totalWaste$"));
        if (!totalWasteItem) {
            throw std::runtime_error("$variable$totalWaste is not found");
        }
        totalWaste = totalWasteItem->getValueAsLong();
        auto passRateItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$passRate$"));
        if (!passRateItem) {
            throw std::runtime_error("$variable$passRate is not found");
        }
        passRate = passRateItem->getValueAsDouble();
        auto isDebugModeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isDebugMode$"));
        if (!isDebugModeItem) {
            throw std::runtime_error("$variable$isDebugMode is not found");
        }
        isDebugMode = isDebugModeItem->getValueAsBool();
        auto isTakePicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isTakePictures$"));
        if (!isTakePicturesItem) {
            throw std::runtime_error("$variable$isTakePictures is not found");
        }
        isTakePictures = isTakePicturesItem->getValueAsBool();
        auto isEliminatingItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isEliminating$"));
        if (!isEliminatingItem) {
            throw std::runtime_error("$variable$isEliminating is not found");
        }
        isEliminating = isEliminatingItem->getValueAsBool();
        auto scrappingRateItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$scrappingRate$"));
        if (!scrappingRateItem) {
            throw std::runtime_error("$variable$scrappingRate is not found");
        }
        scrappingRate = scrappingRateItem->getValueAsBool();
        auto upLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$upLight$"));
        if (!upLightItem) {
            throw std::runtime_error("$variable$upLight is not found");
        }
        upLight = upLightItem->getValueAsBool();
        auto downLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$downLight$"));
        if (!downLightItem) {
            throw std::runtime_error("$variable$downLight is not found");
        }
        downLight = downLightItem->getValueAsBool();
        auto sideLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$sideLight$"));
        if (!sideLightItem) {
            throw std::runtime_error("$variable$sideLight is not found");
        }
        sideLight = sideLightItem->getValueAsBool();
        auto strobeLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$strobeLight$"));
        if (!strobeLightItem) {
            throw std::runtime_error("$variable$strobeLight is not found");
        }
        strobeLight = strobeLightItem->getValueAsBool();
        auto speedItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$speed$"));
        if (!speedItem) {
            throw std::runtime_error("$variable$speed is not found");
        }
        speed = speedItem->getValueAsDouble();
        auto beltSpeedItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$beltSpeed$"));
        if (!beltSpeedItem) {
            throw std::runtime_error("$variable$beltSpeed is not found");
        }
        beltSpeed = beltSpeedItem->getValueAsDouble();
        auto isDefectItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isDefect$"));
        if (!isDefectItem) {
            throw std::runtime_error("$variable$isDefect is not found");
        }
        isDefect = isDefectItem->getValueAsBool();
        auto isPositiveItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isPositive$"));
        if (!isPositiveItem) {
            throw std::runtime_error("$variable$isPositive is not found");
        }
        isPositive = isPositiveItem->getValueAsBool();
    }

    inline ButtonScannerMainWindow::ButtonScannerMainWindow(const ButtonScannerMainWindow& obj)
    {
        totalProduction = obj.totalProduction;
        totalWaste = obj.totalWaste;
        passRate = obj.passRate;
        isDebugMode = obj.isDebugMode;
        isTakePictures = obj.isTakePictures;
        isEliminating = obj.isEliminating;
        scrappingRate = obj.scrappingRate;
        upLight = obj.upLight;
        downLight = obj.downLight;
        sideLight = obj.sideLight;
        strobeLight = obj.strobeLight;
        speed = obj.speed;
        beltSpeed = obj.beltSpeed;
        isDefect = obj.isDefect;
        isPositive = obj.isPositive;
    }

    inline ButtonScannerMainWindow& ButtonScannerMainWindow::operator=(const ButtonScannerMainWindow& obj)
    {
        if (this != &obj) {
            totalProduction = obj.totalProduction;
            totalWaste = obj.totalWaste;
            passRate = obj.passRate;
            isDebugMode = obj.isDebugMode;
            isTakePictures = obj.isTakePictures;
            isEliminating = obj.isEliminating;
            scrappingRate = obj.scrappingRate;
            upLight = obj.upLight;
            downLight = obj.downLight;
            sideLight = obj.sideLight;
            strobeLight = obj.strobeLight;
            speed = obj.speed;
            beltSpeed = obj.beltSpeed;
            isDefect = obj.isDefect;
            isPositive = obj.isPositive;
        }
        return *this;
    }

    inline ButtonScannerMainWindow::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ButtonScannerMainWindow$");
        auto totalProductionItem = std::make_shared<rw::oso::ObjectStoreItem>();
        totalProductionItem->setName("$variable$totalProduction$");
        totalProductionItem->setValueFromLong(totalProduction);
        assembly.addItem(totalProductionItem);
        auto totalWasteItem = std::make_shared<rw::oso::ObjectStoreItem>();
        totalWasteItem->setName("$variable$totalWaste$");
        totalWasteItem->setValueFromLong(totalWaste);
        assembly.addItem(totalWasteItem);
        auto passRateItem = std::make_shared<rw::oso::ObjectStoreItem>();
        passRateItem->setName("$variable$passRate$");
        passRateItem->setValueFromDouble(passRate);
        assembly.addItem(passRateItem);
        auto isDebugModeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isDebugModeItem->setName("$variable$isDebugMode$");
        isDebugModeItem->setValueFromBool(isDebugMode);
        assembly.addItem(isDebugModeItem);
        auto isTakePicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isTakePicturesItem->setName("$variable$isTakePictures$");
        isTakePicturesItem->setValueFromBool(isTakePictures);
        assembly.addItem(isTakePicturesItem);
        auto isEliminatingItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isEliminatingItem->setName("$variable$isEliminating$");
        isEliminatingItem->setValueFromBool(isEliminating);
        assembly.addItem(isEliminatingItem);
        auto scrappingRateItem = std::make_shared<rw::oso::ObjectStoreItem>();
        scrappingRateItem->setName("$variable$scrappingRate$");
        scrappingRateItem->setValueFromBool(scrappingRate);
        assembly.addItem(scrappingRateItem);
        auto upLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        upLightItem->setName("$variable$upLight$");
        upLightItem->setValueFromBool(upLight);
        assembly.addItem(upLightItem);
        auto downLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        downLightItem->setName("$variable$downLight$");
        downLightItem->setValueFromBool(downLight);
        assembly.addItem(downLightItem);
        auto sideLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        sideLightItem->setName("$variable$sideLight$");
        sideLightItem->setValueFromBool(sideLight);
        assembly.addItem(sideLightItem);
        auto strobeLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        strobeLightItem->setName("$variable$strobeLight$");
        strobeLightItem->setValueFromBool(strobeLight);
        assembly.addItem(strobeLightItem);
        auto speedItem = std::make_shared<rw::oso::ObjectStoreItem>();
        speedItem->setName("$variable$speed$");
        speedItem->setValueFromDouble(speed);
        assembly.addItem(speedItem);
        auto beltSpeedItem = std::make_shared<rw::oso::ObjectStoreItem>();
        beltSpeedItem->setName("$variable$beltSpeed$");
        beltSpeedItem->setValueFromDouble(beltSpeed);
        assembly.addItem(beltSpeedItem);
        auto isDefectItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isDefectItem->setName("$variable$isDefect$");
        isDefectItem->setValueFromBool(isDefect);
        assembly.addItem(isDefectItem);
        auto isPositiveItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isPositiveItem->setName("$variable$isPositive$");
        isPositiveItem->setValueFromBool(isPositive);
        assembly.addItem(isPositiveItem);
        return assembly;
    }

    inline bool ButtonScannerMainWindow::operator==(const ButtonScannerMainWindow& obj) const
    {
        return totalProduction == obj.totalProduction && totalWaste == obj.totalWaste && passRate == obj.passRate && isDebugMode == obj.isDebugMode && isTakePictures == obj.isTakePictures && isEliminating == obj.isEliminating && scrappingRate == obj.scrappingRate && upLight == obj.upLight && downLight == obj.downLight && sideLight == obj.sideLight && strobeLight == obj.strobeLight && speed == obj.speed && beltSpeed == obj.beltSpeed && isDefect == obj.isDefect && isPositive == obj.isPositive;
    }

    inline bool ButtonScannerMainWindow::operator!=(const ButtonScannerMainWindow& obj) const
    {
        return !(*this == obj);
    }

}

