#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ButtonScannerDlgWarningManager0
    {
    public:
        ButtonScannerDlgWarningManager0() = default;
        ~ButtonScannerDlgWarningManager0() = default;

        ButtonScannerDlgWarningManager0(const rw::oso::ObjectStoreAssembly& assembly);
        ButtonScannerDlgWarningManager0(const ButtonScannerDlgWarningManager0& obj);

        ButtonScannerDlgWarningManager0& operator=(const ButtonScannerDlgWarningManager0& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ButtonScannerDlgWarningManager0& obj) const;
        bool operator!=(const ButtonScannerDlgWarningManager0& obj) const;

    public:
        bool cameraDisconnect1{ true };
        bool cameraDisconnect2{ true };
        bool cameraDisconnect3{ true };
        bool cameraDisconnect4{ true };
        bool workTrigger1{ true };
        bool workTrigger2{ true };
        bool workTrigger3{ true };
        bool workTrigger4{ true };
        bool airPressure{ true };
    };

    inline ButtonScannerDlgWarningManager0::ButtonScannerDlgWarningManager0(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ButtonScannerDlgWarningManager0$")
        {
            throw std::runtime_error("Assembly is not $class$ButtonScannerDlgWarningManager0$");
        }
        auto cameraDisconnect1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect1$"));
        if (!cameraDisconnect1Item) {
            throw std::runtime_error("$variable$cameraDisconnect1 is not found");
        }
        cameraDisconnect1 = cameraDisconnect1Item->getValueAsBool();
        auto cameraDisconnect2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect2$"));
        if (!cameraDisconnect2Item) {
            throw std::runtime_error("$variable$cameraDisconnect2 is not found");
        }
        cameraDisconnect2 = cameraDisconnect2Item->getValueAsBool();
        auto cameraDisconnect3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect3$"));
        if (!cameraDisconnect3Item) {
            throw std::runtime_error("$variable$cameraDisconnect3 is not found");
        }
        cameraDisconnect3 = cameraDisconnect3Item->getValueAsBool();
        auto cameraDisconnect4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$cameraDisconnect4$"));
        if (!cameraDisconnect4Item) {
            throw std::runtime_error("$variable$cameraDisconnect4 is not found");
        }
        cameraDisconnect4 = cameraDisconnect4Item->getValueAsBool();
        auto workTrigger1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger1$"));
        if (!workTrigger1Item) {
            throw std::runtime_error("$variable$workTrigger1 is not found");
        }
        workTrigger1 = workTrigger1Item->getValueAsBool();
        auto workTrigger2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger2$"));
        if (!workTrigger2Item) {
            throw std::runtime_error("$variable$workTrigger2 is not found");
        }
        workTrigger2 = workTrigger2Item->getValueAsBool();
        auto workTrigger3Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger3$"));
        if (!workTrigger3Item) {
            throw std::runtime_error("$variable$workTrigger3 is not found");
        }
        workTrigger3 = workTrigger3Item->getValueAsBool();
        auto workTrigger4Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$workTrigger4$"));
        if (!workTrigger4Item) {
            throw std::runtime_error("$variable$workTrigger4 is not found");
        }
        workTrigger4 = workTrigger4Item->getValueAsBool();
        auto airPressureItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$airPressure$"));
        if (!airPressureItem) {
            throw std::runtime_error("$variable$airPressure is not found");
        }
        airPressure = airPressureItem->getValueAsBool();
    }

    inline ButtonScannerDlgWarningManager0::ButtonScannerDlgWarningManager0(const ButtonScannerDlgWarningManager0& obj)
    {
        cameraDisconnect1 = obj.cameraDisconnect1;
        cameraDisconnect2 = obj.cameraDisconnect2;
        cameraDisconnect3 = obj.cameraDisconnect3;
        cameraDisconnect4 = obj.cameraDisconnect4;
        workTrigger1 = obj.workTrigger1;
        workTrigger2 = obj.workTrigger2;
        workTrigger3 = obj.workTrigger3;
        workTrigger4 = obj.workTrigger4;
        airPressure = obj.airPressure;
    }

    inline ButtonScannerDlgWarningManager0& ButtonScannerDlgWarningManager0::operator=(const ButtonScannerDlgWarningManager0& obj)
    {
        if (this != &obj) {
            cameraDisconnect1 = obj.cameraDisconnect1;
            cameraDisconnect2 = obj.cameraDisconnect2;
            cameraDisconnect3 = obj.cameraDisconnect3;
            cameraDisconnect4 = obj.cameraDisconnect4;
            workTrigger1 = obj.workTrigger1;
            workTrigger2 = obj.workTrigger2;
            workTrigger3 = obj.workTrigger3;
            workTrigger4 = obj.workTrigger4;
            airPressure = obj.airPressure;
        }
        return *this;
    }

    inline ButtonScannerDlgWarningManager0::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ButtonScannerDlgWarningManager0$");
        auto cameraDisconnect1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        cameraDisconnect1Item->setName("$variable$cameraDisconnect1$");
        cameraDisconnect1Item->setValueFromBool(cameraDisconnect1);
        assembly.addItem(cameraDisconnect1Item);
        auto cameraDisconnect2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        cameraDisconnect2Item->setName("$variable$cameraDisconnect2$");
        cameraDisconnect2Item->setValueFromBool(cameraDisconnect2);
        assembly.addItem(cameraDisconnect2Item);
        auto cameraDisconnect3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        cameraDisconnect3Item->setName("$variable$cameraDisconnect3$");
        cameraDisconnect3Item->setValueFromBool(cameraDisconnect3);
        assembly.addItem(cameraDisconnect3Item);
        auto cameraDisconnect4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        cameraDisconnect4Item->setName("$variable$cameraDisconnect4$");
        cameraDisconnect4Item->setValueFromBool(cameraDisconnect4);
        assembly.addItem(cameraDisconnect4Item);
        auto workTrigger1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        workTrigger1Item->setName("$variable$workTrigger1$");
        workTrigger1Item->setValueFromBool(workTrigger1);
        assembly.addItem(workTrigger1Item);
        auto workTrigger2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        workTrigger2Item->setName("$variable$workTrigger2$");
        workTrigger2Item->setValueFromBool(workTrigger2);
        assembly.addItem(workTrigger2Item);
        auto workTrigger3Item = std::make_shared<rw::oso::ObjectStoreItem>();
        workTrigger3Item->setName("$variable$workTrigger3$");
        workTrigger3Item->setValueFromBool(workTrigger3);
        assembly.addItem(workTrigger3Item);
        auto workTrigger4Item = std::make_shared<rw::oso::ObjectStoreItem>();
        workTrigger4Item->setName("$variable$workTrigger4$");
        workTrigger4Item->setValueFromBool(workTrigger4);
        assembly.addItem(workTrigger4Item);
        auto airPressureItem = std::make_shared<rw::oso::ObjectStoreItem>();
        airPressureItem->setName("$variable$airPressure$");
        airPressureItem->setValueFromBool(airPressure);
        assembly.addItem(airPressureItem);
        return assembly;
    }

    inline bool ButtonScannerDlgWarningManager0::operator==(const ButtonScannerDlgWarningManager0& obj) const
    {
        return cameraDisconnect1 == obj.cameraDisconnect1 && cameraDisconnect2 == obj.cameraDisconnect2 && cameraDisconnect3 == obj.cameraDisconnect3 && cameraDisconnect4 == obj.cameraDisconnect4 && workTrigger1 == obj.workTrigger1 && workTrigger2 == obj.workTrigger2 && workTrigger3 == obj.workTrigger3 && workTrigger4 == obj.workTrigger4 && airPressure == obj.airPressure;
    }

    inline bool ButtonScannerDlgWarningManager0::operator!=(const ButtonScannerDlgWarningManager0& obj) const
    {
        return !(*this == obj);
    }

}

