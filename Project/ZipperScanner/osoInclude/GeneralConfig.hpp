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
        bool qiangGuang{ false };
        bool zhongGuang{ false };
        bool ruoGuang{ false };
        bool isDebug{ false };
        bool isDefect{ false };
        bool isSaveImg{ false };
        bool isshibiekuang{ true };
        bool iswenzi{ true };
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
        auto qiangGuangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiangGuang$"));
        if (!qiangGuangItem) {
            throw std::runtime_error("$variable$qiangGuang is not found");
        }
        qiangGuang = qiangGuangItem->getValueAsBool();
        auto zhongGuangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zhongGuang$"));
        if (!zhongGuangItem) {
            throw std::runtime_error("$variable$zhongGuang is not found");
        }
        zhongGuang = zhongGuangItem->getValueAsBool();
        auto ruoGuangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$ruoGuang$"));
        if (!ruoGuangItem) {
            throw std::runtime_error("$variable$ruoGuang is not found");
        }
        ruoGuang = ruoGuangItem->getValueAsBool();
        auto isDebugItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isDebug$"));
        if (!isDebugItem) {
            throw std::runtime_error("$variable$isDebug is not found");
        }
        isDebug = isDebugItem->getValueAsBool();
        auto isDefectItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isDefect$"));
        if (!isDefectItem) {
            throw std::runtime_error("$variable$isDefect is not found");
        }
        isDefect = isDefectItem->getValueAsBool();
        auto isSaveImgItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isSaveImg$"));
        if (!isSaveImgItem) {
            throw std::runtime_error("$variable$isSaveImg is not found");
        }
        isSaveImg = isSaveImgItem->getValueAsBool();
        auto isshibiekuangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isshibiekuang$"));
        if (!isshibiekuangItem) {
            throw std::runtime_error("$variable$isshibiekuang is not found");
        }
        isshibiekuang = isshibiekuangItem->getValueAsBool();
        auto iswenziItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$iswenzi$"));
        if (!iswenziItem) {
            throw std::runtime_error("$variable$iswenzi is not found");
        }
        iswenzi = iswenziItem->getValueAsBool();
    }

    inline GeneralConfig::GeneralConfig(const GeneralConfig& obj)
    {
        totalProductionVolume = obj.totalProductionVolume;
        totalDefectiveVolume = obj.totalDefectiveVolume;
        productionYield = obj.productionYield;
        qiangGuang = obj.qiangGuang;
        zhongGuang = obj.zhongGuang;
        ruoGuang = obj.ruoGuang;
        isDebug = obj.isDebug;
        isDefect = obj.isDefect;
        isSaveImg = obj.isSaveImg;
        isshibiekuang = obj.isshibiekuang;
        iswenzi = obj.iswenzi;
    }

    inline GeneralConfig& GeneralConfig::operator=(const GeneralConfig& obj)
    {
        if (this != &obj) {
            totalProductionVolume = obj.totalProductionVolume;
            totalDefectiveVolume = obj.totalDefectiveVolume;
            productionYield = obj.productionYield;
            qiangGuang = obj.qiangGuang;
            zhongGuang = obj.zhongGuang;
            ruoGuang = obj.ruoGuang;
            isDebug = obj.isDebug;
            isDefect = obj.isDefect;
            isSaveImg = obj.isSaveImg;
            isshibiekuang = obj.isshibiekuang;
            iswenzi = obj.iswenzi;
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
        auto qiangGuangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiangGuangItem->setName("$variable$qiangGuang$");
        qiangGuangItem->setValueFromBool(qiangGuang);
        assembly.addItem(qiangGuangItem);
        auto zhongGuangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zhongGuangItem->setName("$variable$zhongGuang$");
        zhongGuangItem->setValueFromBool(zhongGuang);
        assembly.addItem(zhongGuangItem);
        auto ruoGuangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        ruoGuangItem->setName("$variable$ruoGuang$");
        ruoGuangItem->setValueFromBool(ruoGuang);
        assembly.addItem(ruoGuangItem);
        auto isDebugItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isDebugItem->setName("$variable$isDebug$");
        isDebugItem->setValueFromBool(isDebug);
        assembly.addItem(isDebugItem);
        auto isDefectItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isDefectItem->setName("$variable$isDefect$");
        isDefectItem->setValueFromBool(isDefect);
        assembly.addItem(isDefectItem);
        auto isSaveImgItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isSaveImgItem->setName("$variable$isSaveImg$");
        isSaveImgItem->setValueFromBool(isSaveImg);
        assembly.addItem(isSaveImgItem);
        auto isshibiekuangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isshibiekuangItem->setName("$variable$isshibiekuang$");
        isshibiekuangItem->setValueFromBool(isshibiekuang);
        assembly.addItem(isshibiekuangItem);
        auto iswenziItem = std::make_shared<rw::oso::ObjectStoreItem>();
        iswenziItem->setName("$variable$iswenzi$");
        iswenziItem->setValueFromBool(iswenzi);
        assembly.addItem(iswenziItem);
        return assembly;
    }

    inline bool GeneralConfig::operator==(const GeneralConfig& obj) const
    {
        return totalProductionVolume == obj.totalProductionVolume && totalDefectiveVolume == obj.totalDefectiveVolume && productionYield == obj.productionYield && qiangGuang == obj.qiangGuang && zhongGuang == obj.zhongGuang && ruoGuang == obj.ruoGuang && isDebug == obj.isDebug && isDefect == obj.isDefect && isSaveImg == obj.isSaveImg && isshibiekuang == obj.isshibiekuang && iswenzi == obj.iswenzi;
    }

    inline bool GeneralConfig::operator!=(const GeneralConfig& obj) const
    {
        return !(*this == obj);
    }

}

