#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class GeneralConfig
    {
    public:
        GeneralConfig() = default;
        ~GeneralConfig() = default;

        GeneralConfig(const rw::oso::ObjectStoreAssembly& assembly);
        GeneralConfig(const GeneralConfig& obj);

        GeneralConfig& operator=(const GeneralConfig& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const GeneralConfig& obj) const;
        bool operator!=(const GeneralConfig& obj) const;

    public:
        int totalProductionVolume{ 0 };
        int totalDefectiveVolume{ 0 };
        double productionYield{ 0.0 };
    };

    inline GeneralConfig::GeneralConfig(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$GeneralConfig$")
        {
            throw std::runtime_error("Assembly is not $class$GeneralConfig$");
        }
        auto totalProductionVolumeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$totalProductionVolume$"));
        if (!totalProductionVolumeItem) {
            throw std::runtime_error("$variable$totalProductionVolume is not found");
        }
        totalProductionVolume = totalProductionVolumeItem->getValueAsInt();
        auto totalDefectiveVolumeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$totalDefectiveVolume$"));
        if (!totalDefectiveVolumeItem) {
            throw std::runtime_error("$variable$totalDefectiveVolume is not found");
        }
        totalDefectiveVolume = totalDefectiveVolumeItem->getValueAsInt();
        auto productionYieldItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$productionYield$"));
        if (!productionYieldItem) {
            throw std::runtime_error("$variable$productionYield is not found");
        }
        productionYield = productionYieldItem->getValueAsDouble();
    }

    inline GeneralConfig::GeneralConfig(const GeneralConfig& obj)
    {
        totalProductionVolume = obj.totalProductionVolume;
        totalDefectiveVolume = obj.totalDefectiveVolume;
        productionYield = obj.productionYield;
    }

    inline GeneralConfig& GeneralConfig::operator=(const GeneralConfig& obj)
    {
        if (this != &obj) {
            totalProductionVolume = obj.totalProductionVolume;
            totalDefectiveVolume = obj.totalDefectiveVolume;
            productionYield = obj.productionYield;
        }
        return *this;
    }

    inline GeneralConfig::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$GeneralConfig$");
        auto totalProductionVolumeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        totalProductionVolumeItem->setName("$variable$totalProductionVolume$");
        totalProductionVolumeItem->setValueFromInt(totalProductionVolume);
        assembly.addItem(totalProductionVolumeItem);
        auto totalDefectiveVolumeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        totalDefectiveVolumeItem->setName("$variable$totalDefectiveVolume$");
        totalDefectiveVolumeItem->setValueFromInt(totalDefectiveVolume);
        assembly.addItem(totalDefectiveVolumeItem);
        auto productionYieldItem = std::make_shared<rw::oso::ObjectStoreItem>();
        productionYieldItem->setName("$variable$productionYield$");
        productionYieldItem->setValueFromDouble(productionYield);
        assembly.addItem(productionYieldItem);
        return assembly;
    }

    inline bool GeneralConfig::operator==(const GeneralConfig& obj) const
    {
        return totalProductionVolume == obj.totalProductionVolume && totalDefectiveVolume == obj.totalDefectiveVolume && productionYield == obj.productionYield;
    }

    inline bool GeneralConfig::operator!=(const GeneralConfig& obj) const
    {
        return !(*this == obj);
    }

}

