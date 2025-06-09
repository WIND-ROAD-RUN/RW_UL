#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ZipperScannerDlgExposureTimeSet
    {
    public:
        ZipperScannerDlgExposureTimeSet() = default;
        ~ZipperScannerDlgExposureTimeSet() = default;

        ZipperScannerDlgExposureTimeSet(const rw::oso::ObjectStoreAssembly& assembly);
        ZipperScannerDlgExposureTimeSet(const ZipperScannerDlgExposureTimeSet& obj);

        ZipperScannerDlgExposureTimeSet& operator=(const ZipperScannerDlgExposureTimeSet& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ZipperScannerDlgExposureTimeSet& obj) const;
        bool operator!=(const ZipperScannerDlgExposureTimeSet& obj) const;

    public:
        size_t exposureTime{ 1000 };
    };

    inline ZipperScannerDlgExposureTimeSet::ZipperScannerDlgExposureTimeSet(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ZipperScannerDlgExposureTimeSet$")
        {
            throw std::runtime_error("Assembly is not $class$ZipperScannerDlgExposureTimeSet$");
        }
        auto exposureTimeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$exposureTime$"));
        if (!exposureTimeItem) {
            throw std::runtime_error("$variable$exposureTime is not found");
        }
        exposureTime = exposureTimeItem->getValueAsLong();
    }

    inline ZipperScannerDlgExposureTimeSet::ZipperScannerDlgExposureTimeSet(const ZipperScannerDlgExposureTimeSet& obj)
    {
        exposureTime = obj.exposureTime;
    }

    inline ZipperScannerDlgExposureTimeSet& ZipperScannerDlgExposureTimeSet::operator=(const ZipperScannerDlgExposureTimeSet& obj)
    {
        if (this != &obj) {
            exposureTime = obj.exposureTime;
        }
        return *this;
    }

    inline ZipperScannerDlgExposureTimeSet::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ZipperScannerDlgExposureTimeSet$");
        auto exposureTimeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        exposureTimeItem->setName("$variable$exposureTime$");
        exposureTimeItem->setValueFromLong(exposureTime);
        assembly.addItem(exposureTimeItem);
        return assembly;
    }

    inline bool ZipperScannerDlgExposureTimeSet::operator==(const ZipperScannerDlgExposureTimeSet& obj) const
    {
        return exposureTime == obj.exposureTime;
    }

    inline bool ZipperScannerDlgExposureTimeSet::operator!=(const ZipperScannerDlgExposureTimeSet& obj) const
    {
        return !(*this == obj);
    }

}

