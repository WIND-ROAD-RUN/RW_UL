#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ButtonScannerDlgExposureTimeSet
    {
    public:
        ButtonScannerDlgExposureTimeSet() = default;
        ~ButtonScannerDlgExposureTimeSet() = default;

        ButtonScannerDlgExposureTimeSet(const rw::oso::ObjectStoreAssembly& assembly);
        ButtonScannerDlgExposureTimeSet(const ButtonScannerDlgExposureTimeSet& obj);

        ButtonScannerDlgExposureTimeSet& operator=(const ButtonScannerDlgExposureTimeSet& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ButtonScannerDlgExposureTimeSet& obj) const;
        bool operator!=(const ButtonScannerDlgExposureTimeSet& obj) const;

    public:
        int expousureTime{ 0 };
    };

    inline ButtonScannerDlgExposureTimeSet::ButtonScannerDlgExposureTimeSet(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ButtonScannerDlgExposureTimeSet$")
        {
            throw std::runtime_error("Assembly is not $class$ButtonScannerDlgExposureTimeSet$");
        }
        auto expousureTimeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$expousureTime$"));
        if (!expousureTimeItem) {
            throw std::runtime_error("$variable$expousureTime is not found");
        }
        expousureTime = expousureTimeItem->getValueAsInt();
    }

    inline ButtonScannerDlgExposureTimeSet::ButtonScannerDlgExposureTimeSet(const ButtonScannerDlgExposureTimeSet& obj)
    {
        expousureTime = obj.expousureTime;
    }

    inline ButtonScannerDlgExposureTimeSet& ButtonScannerDlgExposureTimeSet::operator=(const ButtonScannerDlgExposureTimeSet& obj)
    {
        if (this != &obj) {
            expousureTime = obj.expousureTime;
        }
        return *this;
    }

    inline ButtonScannerDlgExposureTimeSet::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ButtonScannerDlgExposureTimeSet$");
        auto expousureTimeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        expousureTimeItem->setName("$variable$expousureTime$");
        expousureTimeItem->setValueFromInt(expousureTime);
        assembly.addItem(expousureTimeItem);
        return assembly;
    }

    inline bool ButtonScannerDlgExposureTimeSet::operator==(const ButtonScannerDlgExposureTimeSet& obj) const
    {
        return expousureTime == obj.expousureTime;
    }

    inline bool ButtonScannerDlgExposureTimeSet::operator!=(const ButtonScannerDlgExposureTimeSet& obj) const
    {
        return !(*this == obj);
    }

}

