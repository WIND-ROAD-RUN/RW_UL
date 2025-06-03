#pragma once

#include"oso_core.h"
#include <string>
#include <vector>

namespace cdm {
    class WarningIOSetConfig
    {
    public:
        WarningIOSetConfig() = default;
        ~WarningIOSetConfig() = default;

        WarningIOSetConfig(const rw::oso::ObjectStoreAssembly& assembly);
        WarningIOSetConfig(const WarningIOSetConfig& obj);

        WarningIOSetConfig& operator=(const WarningIOSetConfig& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const WarningIOSetConfig& obj) const;
        bool operator!=(const WarningIOSetConfig& obj) const;

    public:
        int DIStart{ 0 };
        int DIStop{ 0 };
        int DIShutdownComputer{ 0 };
        int DIAirPressure{ 0 };
        int DICameraTrigger1{ 0 };
        int DICameraTrigger2{ 0 };
        int DICameraTrigger3{ 0 };
        int DICameraTrigger4{ 0 };
        int DOMotoPower{ 0 };
        int DOBlow1{ 0 };
        int DOBlow2{ 0 };
        int DOBlow3{ 0 };
        int DOBlow4{ 0 };
        int DOGreenLight{ 0 };
        int DORedLight{ 0 };
        int DOUpLight{ 0 };
        int DOSideLight{ 0 };
        int DODownLight{ 0 };
        int DOStrobeLight{ 0 };
        int DOStartBelt{ 0 };
    };

    inline WarningIOSetConfig::WarningIOSetConfig(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$WarningIOSetConfig$")
        {
            throw std::runtime_error("Assembly is not $class$WarningIOSetConfig$");
        }
        auto DIStartItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DIStart$"));
        if (!DIStartItem) {
            throw std::runtime_error("$variable$DIStart is not found");
        }
        DIStart = DIStartItem->getValueAsInt();
        auto DIStopItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DIStop$"));
        if (!DIStopItem) {
            throw std::runtime_error("$variable$DIStop is not found");
        }
        DIStop = DIStopItem->getValueAsInt();
        auto DIShutdownComputerItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DIShutdownComputer$"));
        if (!DIShutdownComputerItem) {
            throw std::runtime_error("$variable$DIShutdownComputer is not found");
        }
        DIShutdownComputer = DIShutdownComputerItem->getValueAsInt();
        auto DIAirPressureItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DIAirPressure$"));
        if (!DIAirPressureItem) {
            throw std::runtime_error("$variable$DIAirPressure is not found");
        }
        DIAirPressure = DIAirPressureItem->getValueAsInt();
        auto DICameraTrigger1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DICameraTrigger1$"));
        if (!DICameraTrigger1Item) {
            throw std::runtime_error("$variable$DICameraTrigger1 is not found");
        }
        DICameraTrigger1 = DICameraTrigger1Item->getValueAsInt();
        auto DICameraTrigger2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DICameraTrigger2$"));
        if (!DICameraTrigger2Item) {
            throw std::runtime_error("$variable$DICameraTrigger2 is not found");
        }
        DICameraTrigger2 = DICameraTrigger2Item->getValueAsInt();
        auto DICameraTrigger3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DICameraTrigger3$"));
        if (!DICameraTrigger3Item) {
            throw std::runtime_error("$variable$DICameraTrigger3 is not found");
        }
        DICameraTrigger3 = DICameraTrigger3Item->getValueAsInt();
        auto DICameraTrigger4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DICameraTrigger4$"));
        if (!DICameraTrigger4Item) {
            throw std::runtime_error("$variable$DICameraTrigger4 is not found");
        }
        DICameraTrigger4 = DICameraTrigger4Item->getValueAsInt();
        auto DOMotoPowerItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOMotoPower$"));
        if (!DOMotoPowerItem) {
            throw std::runtime_error("$variable$DOMotoPower is not found");
        }
        DOMotoPower = DOMotoPowerItem->getValueAsInt();
        auto DOBlow1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOBlow1$"));
        if (!DOBlow1Item) {
            throw std::runtime_error("$variable$DOBlow1 is not found");
        }
        DOBlow1 = DOBlow1Item->getValueAsInt();
        auto DOBlow2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOBlow2$"));
        if (!DOBlow2Item) {
            throw std::runtime_error("$variable$DOBlow2 is not found");
        }
        DOBlow2 = DOBlow2Item->getValueAsInt();
        auto DOBlow3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOBlow3$"));
        if (!DOBlow3Item) {
            throw std::runtime_error("$variable$DOBlow3 is not found");
        }
        DOBlow3 = DOBlow3Item->getValueAsInt();
        auto DOBlow4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOBlow4$"));
        if (!DOBlow4Item) {
            throw std::runtime_error("$variable$DOBlow4 is not found");
        }
        DOBlow4 = DOBlow4Item->getValueAsInt();
        auto DOGreenLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOGreenLight$"));
        if (!DOGreenLightItem) {
            throw std::runtime_error("$variable$DOGreenLight is not found");
        }
        DOGreenLight = DOGreenLightItem->getValueAsInt();
        auto DORedLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DORedLight$"));
        if (!DORedLightItem) {
            throw std::runtime_error("$variable$DORedLight is not found");
        }
        DORedLight = DORedLightItem->getValueAsInt();
        auto DOUpLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOUpLight$"));
        if (!DOUpLightItem) {
            throw std::runtime_error("$variable$DOUpLight is not found");
        }
        DOUpLight = DOUpLightItem->getValueAsInt();
        auto DOSideLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOSideLight$"));
        if (!DOSideLightItem) {
            throw std::runtime_error("$variable$DOSideLight is not found");
        }
        DOSideLight = DOSideLightItem->getValueAsInt();
        auto DODownLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DODownLight$"));
        if (!DODownLightItem) {
            throw std::runtime_error("$variable$DODownLight is not found");
        }
        DODownLight = DODownLightItem->getValueAsInt();
        auto DOStrobeLightItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOStrobeLight$"));
        if (!DOStrobeLightItem) {
            throw std::runtime_error("$variable$DOStrobeLight is not found");
        }
        DOStrobeLight = DOStrobeLightItem->getValueAsInt();
        auto DOStartBeltItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$DOStartBelt$"));
        if (!DOStartBeltItem) {
            throw std::runtime_error("$variable$DOStartBelt is not found");
        }
        DOStartBelt = DOStartBeltItem->getValueAsInt();
    }

    inline WarningIOSetConfig::WarningIOSetConfig(const WarningIOSetConfig& obj)
    {
        DIStart = obj.DIStart;
        DIStop = obj.DIStop;
        DIShutdownComputer = obj.DIShutdownComputer;
        DIAirPressure = obj.DIAirPressure;
        DICameraTrigger1 = obj.DICameraTrigger1;
        DICameraTrigger2 = obj.DICameraTrigger2;
        DICameraTrigger3 = obj.DICameraTrigger3;
        DICameraTrigger4 = obj.DICameraTrigger4;
        DOMotoPower = obj.DOMotoPower;
        DOBlow1 = obj.DOBlow1;
        DOBlow2 = obj.DOBlow2;
        DOBlow3 = obj.DOBlow3;
        DOBlow4 = obj.DOBlow4;
        DOGreenLight = obj.DOGreenLight;
        DORedLight = obj.DORedLight;
        DOUpLight = obj.DOUpLight;
        DOSideLight = obj.DOSideLight;
        DODownLight = obj.DODownLight;
        DOStrobeLight = obj.DOStrobeLight;
        DOStartBelt = obj.DOStartBelt;
    }

    inline WarningIOSetConfig& WarningIOSetConfig::operator=(const WarningIOSetConfig& obj)
    {
        if (this != &obj) {
            DIStart = obj.DIStart;
            DIStop = obj.DIStop;
            DIShutdownComputer = obj.DIShutdownComputer;
            DIAirPressure = obj.DIAirPressure;
            DICameraTrigger1 = obj.DICameraTrigger1;
            DICameraTrigger2 = obj.DICameraTrigger2;
            DICameraTrigger3 = obj.DICameraTrigger3;
            DICameraTrigger4 = obj.DICameraTrigger4;
            DOMotoPower = obj.DOMotoPower;
            DOBlow1 = obj.DOBlow1;
            DOBlow2 = obj.DOBlow2;
            DOBlow3 = obj.DOBlow3;
            DOBlow4 = obj.DOBlow4;
            DOGreenLight = obj.DOGreenLight;
            DORedLight = obj.DORedLight;
            DOUpLight = obj.DOUpLight;
            DOSideLight = obj.DOSideLight;
            DODownLight = obj.DODownLight;
            DOStrobeLight = obj.DOStrobeLight;
            DOStartBelt = obj.DOStartBelt;
        }
        return *this;
    }

    inline WarningIOSetConfig::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$WarningIOSetConfig$");
        auto DIStartItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DIStartItem->setName("$variable$DIStart$");
        DIStartItem->setValueFromInt(DIStart);
        assembly.addItem(DIStartItem);
        auto DIStopItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DIStopItem->setName("$variable$DIStop$");
        DIStopItem->setValueFromInt(DIStop);
        assembly.addItem(DIStopItem);
        auto DIShutdownComputerItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DIShutdownComputerItem->setName("$variable$DIShutdownComputer$");
        DIShutdownComputerItem->setValueFromInt(DIShutdownComputer);
        assembly.addItem(DIShutdownComputerItem);
        auto DIAirPressureItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DIAirPressureItem->setName("$variable$DIAirPressure$");
        DIAirPressureItem->setValueFromInt(DIAirPressure);
        assembly.addItem(DIAirPressureItem);
        auto DICameraTrigger1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DICameraTrigger1Item->setName("$variable$DICameraTrigger1$");
        DICameraTrigger1Item->setValueFromInt(DICameraTrigger1);
        assembly.addItem(DICameraTrigger1Item);
        auto DICameraTrigger2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DICameraTrigger2Item->setName("$variable$DICameraTrigger2$");
        DICameraTrigger2Item->setValueFromInt(DICameraTrigger2);
        assembly.addItem(DICameraTrigger2Item);
        auto DICameraTrigger3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DICameraTrigger3Item->setName("$variable$DICameraTrigger3$");
        DICameraTrigger3Item->setValueFromInt(DICameraTrigger3);
        assembly.addItem(DICameraTrigger3Item);
        auto DICameraTrigger4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DICameraTrigger4Item->setName("$variable$DICameraTrigger4$");
        DICameraTrigger4Item->setValueFromInt(DICameraTrigger4);
        assembly.addItem(DICameraTrigger4Item);
        auto DOMotoPowerItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DOMotoPowerItem->setName("$variable$DOMotoPower$");
        DOMotoPowerItem->setValueFromInt(DOMotoPower);
        assembly.addItem(DOMotoPowerItem);
        auto DOBlow1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DOBlow1Item->setName("$variable$DOBlow1$");
        DOBlow1Item->setValueFromInt(DOBlow1);
        assembly.addItem(DOBlow1Item);
        auto DOBlow2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DOBlow2Item->setName("$variable$DOBlow2$");
        DOBlow2Item->setValueFromInt(DOBlow2);
        assembly.addItem(DOBlow2Item);
        auto DOBlow3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DOBlow3Item->setName("$variable$DOBlow3$");
        DOBlow3Item->setValueFromInt(DOBlow3);
        assembly.addItem(DOBlow3Item);
        auto DOBlow4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        DOBlow4Item->setName("$variable$DOBlow4$");
        DOBlow4Item->setValueFromInt(DOBlow4);
        assembly.addItem(DOBlow4Item);
        auto DOGreenLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DOGreenLightItem->setName("$variable$DOGreenLight$");
        DOGreenLightItem->setValueFromInt(DOGreenLight);
        assembly.addItem(DOGreenLightItem);
        auto DORedLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DORedLightItem->setName("$variable$DORedLight$");
        DORedLightItem->setValueFromInt(DORedLight);
        assembly.addItem(DORedLightItem);
        auto DOUpLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DOUpLightItem->setName("$variable$DOUpLight$");
        DOUpLightItem->setValueFromInt(DOUpLight);
        assembly.addItem(DOUpLightItem);
        auto DOSideLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DOSideLightItem->setName("$variable$DOSideLight$");
        DOSideLightItem->setValueFromInt(DOSideLight);
        assembly.addItem(DOSideLightItem);
        auto DODownLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DODownLightItem->setName("$variable$DODownLight$");
        DODownLightItem->setValueFromInt(DODownLight);
        assembly.addItem(DODownLightItem);
        auto DOStrobeLightItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DOStrobeLightItem->setName("$variable$DOStrobeLight$");
        DOStrobeLightItem->setValueFromInt(DOStrobeLight);
        assembly.addItem(DOStrobeLightItem);
        auto DOStartBeltItem = std::make_shared<rw::oso::ObjectStoreItem>();
        DOStartBeltItem->setName("$variable$DOStartBelt$");
        DOStartBeltItem->setValueFromInt(DOStartBelt);
        assembly.addItem(DOStartBeltItem);
        return assembly;
    }

    inline bool WarningIOSetConfig::operator==(const WarningIOSetConfig& obj) const
    {
        return DIStart == obj.DIStart && DIStop == obj.DIStop && DIShutdownComputer == obj.DIShutdownComputer && DIAirPressure == obj.DIAirPressure && DICameraTrigger1 == obj.DICameraTrigger1 && DICameraTrigger2 == obj.DICameraTrigger2 && DICameraTrigger3 == obj.DICameraTrigger3 && DICameraTrigger4 == obj.DICameraTrigger4 && DOMotoPower == obj.DOMotoPower && DOBlow1 == obj.DOBlow1 && DOBlow2 == obj.DOBlow2 && DOBlow3 == obj.DOBlow3 && DOBlow4 == obj.DOBlow4 && DOGreenLight == obj.DOGreenLight && DORedLight == obj.DORedLight && DOUpLight == obj.DOUpLight && DOSideLight == obj.DOSideLight && DODownLight == obj.DODownLight && DOStrobeLight == obj.DOStrobeLight && DOStartBelt == obj.DOStartBelt;
    }

    inline bool WarningIOSetConfig::operator!=(const WarningIOSetConfig& obj) const
    {
        return !(*this == obj);
    }

}

