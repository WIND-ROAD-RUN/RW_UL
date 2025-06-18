#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class GeneralConfigSmartCroppingOfBags
    {
    public:
        GeneralConfigSmartCroppingOfBags() = default;
        ~GeneralConfigSmartCroppingOfBags() = default;

        GeneralConfigSmartCroppingOfBags(const rw::oso::ObjectStoreAssembly& assembly);
        GeneralConfigSmartCroppingOfBags(const GeneralConfigSmartCroppingOfBags& obj);

        GeneralConfigSmartCroppingOfBags& operator=(const GeneralConfigSmartCroppingOfBags& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const GeneralConfigSmartCroppingOfBags& obj) const;
        bool operator!=(const GeneralConfigSmartCroppingOfBags& obj) const;

    public:
        bool iszhinengcaiqie{ false };
        int shengchanzongliang{ 0 };
        double shengchanlianglv{ 0 };
        int feipinshuliang{ 0 };
        double pingjundaichang{ 0 };
        bool istifei{ false };
        bool isDebug{ false };
        bool iscuntu{ false };
        bool isyinshuajiance{ false };
        double liangdu{ 0 };
        int daizizhonglei{ 0 };
        double baoguang{ 0 };
    };

    inline GeneralConfigSmartCroppingOfBags::GeneralConfigSmartCroppingOfBags(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$GeneralConfigSmartCroppingOfBags$")
        {
            throw std::runtime_error("Assembly is not $class$GeneralConfigSmartCroppingOfBags$");
        }
        auto iszhinengcaiqieItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$iszhinengcaiqie$"));
        if (!iszhinengcaiqieItem) {
            throw std::runtime_error("$variable$iszhinengcaiqie is not found");
        }
        iszhinengcaiqie = iszhinengcaiqieItem->getValueAsBool();
        auto shengchanzongliangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shengchanzongliang$"));
        if (!shengchanzongliangItem) {
            throw std::runtime_error("$variable$shengchanzongliang is not found");
        }
        shengchanzongliang = shengchanzongliangItem->getValueAsInt();
        auto shengchanlianglvItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shengchanlianglv$"));
        if (!shengchanlianglvItem) {
            throw std::runtime_error("$variable$shengchanlianglv is not found");
        }
        shengchanlianglv = shengchanlianglvItem->getValueAsDouble();
        auto feipinshuliangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$feipinshuliang$"));
        if (!feipinshuliangItem) {
            throw std::runtime_error("$variable$feipinshuliang is not found");
        }
        feipinshuliang = feipinshuliangItem->getValueAsInt();
        auto pingjundaichangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pingjundaichang$"));
        if (!pingjundaichangItem) {
            throw std::runtime_error("$variable$pingjundaichang is not found");
        }
        pingjundaichang = pingjundaichangItem->getValueAsDouble();
        auto istifeiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$istifei$"));
        if (!istifeiItem) {
            throw std::runtime_error("$variable$istifei is not found");
        }
        istifei = istifeiItem->getValueAsBool();
        auto isDebugItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isDebug$"));
        if (!isDebugItem) {
            throw std::runtime_error("$variable$isDebug is not found");
        }
        isDebug = isDebugItem->getValueAsBool();
        auto iscuntuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$iscuntu$"));
        if (!iscuntuItem) {
            throw std::runtime_error("$variable$iscuntu is not found");
        }
        iscuntu = iscuntuItem->getValueAsBool();
        auto isyinshuajianceItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isyinshuajiance$"));
        if (!isyinshuajianceItem) {
            throw std::runtime_error("$variable$isyinshuajiance is not found");
        }
        isyinshuajiance = isyinshuajianceItem->getValueAsBool();
        auto liangduItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$liangdu$"));
        if (!liangduItem) {
            throw std::runtime_error("$variable$liangdu is not found");
        }
        liangdu = liangduItem->getValueAsDouble();
        auto daizizhongleiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$daizizhonglei$"));
        if (!daizizhongleiItem) {
            throw std::runtime_error("$variable$daizizhonglei is not found");
        }
        daizizhonglei = daizizhongleiItem->getValueAsInt();
        auto baoguangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baoguang$"));
        if (!baoguangItem) {
            throw std::runtime_error("$variable$baoguang is not found");
        }
        baoguang = baoguangItem->getValueAsDouble();
    }

    inline GeneralConfigSmartCroppingOfBags::GeneralConfigSmartCroppingOfBags(const GeneralConfigSmartCroppingOfBags& obj)
    {
        iszhinengcaiqie = obj.iszhinengcaiqie;
        shengchanzongliang = obj.shengchanzongliang;
        shengchanlianglv = obj.shengchanlianglv;
        feipinshuliang = obj.feipinshuliang;
        pingjundaichang = obj.pingjundaichang;
        istifei = obj.istifei;
        isDebug = obj.isDebug;
        iscuntu = obj.iscuntu;
        isyinshuajiance = obj.isyinshuajiance;
        liangdu = obj.liangdu;
        daizizhonglei = obj.daizizhonglei;
        baoguang = obj.baoguang;
    }

    inline GeneralConfigSmartCroppingOfBags& GeneralConfigSmartCroppingOfBags::operator=(const GeneralConfigSmartCroppingOfBags& obj)
    {
        if (this != &obj) {
            iszhinengcaiqie = obj.iszhinengcaiqie;
            shengchanzongliang = obj.shengchanzongliang;
            shengchanlianglv = obj.shengchanlianglv;
            feipinshuliang = obj.feipinshuliang;
            pingjundaichang = obj.pingjundaichang;
            istifei = obj.istifei;
            isDebug = obj.isDebug;
            iscuntu = obj.iscuntu;
            isyinshuajiance = obj.isyinshuajiance;
            liangdu = obj.liangdu;
            daizizhonglei = obj.daizizhonglei;
            baoguang = obj.baoguang;
        }
        return *this;
    }

    inline GeneralConfigSmartCroppingOfBags::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$GeneralConfigSmartCroppingOfBags$");
        auto iszhinengcaiqieItem = std::make_shared<rw::oso::ObjectStoreItem>();
        iszhinengcaiqieItem->setName("$variable$iszhinengcaiqie$");
        iszhinengcaiqieItem->setValueFromBool(iszhinengcaiqie);
        assembly.addItem(iszhinengcaiqieItem);
        auto shengchanzongliangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shengchanzongliangItem->setName("$variable$shengchanzongliang$");
        shengchanzongliangItem->setValueFromInt(shengchanzongliang);
        assembly.addItem(shengchanzongliangItem);
        auto shengchanlianglvItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shengchanlianglvItem->setName("$variable$shengchanlianglv$");
        shengchanlianglvItem->setValueFromDouble(shengchanlianglv);
        assembly.addItem(shengchanlianglvItem);
        auto feipinshuliangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        feipinshuliangItem->setName("$variable$feipinshuliang$");
        feipinshuliangItem->setValueFromInt(feipinshuliang);
        assembly.addItem(feipinshuliangItem);
        auto pingjundaichangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        pingjundaichangItem->setName("$variable$pingjundaichang$");
        pingjundaichangItem->setValueFromDouble(pingjundaichang);
        assembly.addItem(pingjundaichangItem);
        auto istifeiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        istifeiItem->setName("$variable$istifei$");
        istifeiItem->setValueFromBool(istifei);
        assembly.addItem(istifeiItem);
        auto isDebugItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isDebugItem->setName("$variable$isDebug$");
        isDebugItem->setValueFromBool(isDebug);
        assembly.addItem(isDebugItem);
        auto iscuntuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        iscuntuItem->setName("$variable$iscuntu$");
        iscuntuItem->setValueFromBool(iscuntu);
        assembly.addItem(iscuntuItem);
        auto isyinshuajianceItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isyinshuajianceItem->setName("$variable$isyinshuajiance$");
        isyinshuajianceItem->setValueFromBool(isyinshuajiance);
        assembly.addItem(isyinshuajianceItem);
        auto liangduItem = std::make_shared<rw::oso::ObjectStoreItem>();
        liangduItem->setName("$variable$liangdu$");
        liangduItem->setValueFromDouble(liangdu);
        assembly.addItem(liangduItem);
        auto daizizhongleiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        daizizhongleiItem->setName("$variable$daizizhonglei$");
        daizizhongleiItem->setValueFromInt(daizizhonglei);
        assembly.addItem(daizizhongleiItem);
        auto baoguangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        baoguangItem->setName("$variable$baoguang$");
        baoguangItem->setValueFromDouble(baoguang);
        assembly.addItem(baoguangItem);
        return assembly;
    }

    inline bool GeneralConfigSmartCroppingOfBags::operator==(const GeneralConfigSmartCroppingOfBags& obj) const
    {
        return iszhinengcaiqie == obj.iszhinengcaiqie && shengchanzongliang == obj.shengchanzongliang && shengchanlianglv == obj.shengchanlianglv && feipinshuliang == obj.feipinshuliang && pingjundaichang == obj.pingjundaichang && istifei == obj.istifei && isDebug == obj.isDebug && iscuntu == obj.iscuntu && isyinshuajiance == obj.isyinshuajiance && liangdu == obj.liangdu && daizizhonglei == obj.daizizhonglei && baoguang == obj.baoguang;
    }

    inline bool GeneralConfigSmartCroppingOfBags::operator!=(const GeneralConfigSmartCroppingOfBags& obj) const
    {
        return !(*this == obj);
    }

}

