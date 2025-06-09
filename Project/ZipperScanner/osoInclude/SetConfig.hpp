#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class SetConfig
    {
    public:
        SetConfig() = default;
        ~SetConfig() = default;

        SetConfig(const rw::oso::ObjectStoreAssembly& assembly);
        SetConfig(const SetConfig& obj);

        SetConfig& operator=(const SetConfig& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const SetConfig& obj) const;
        bool operator!=(const SetConfig& obj) const;

    public:
        double tiFeiChiXuShiJian1{ 0 };
        double yanChiTiFeiShiJian1{ 0 };
        double tiFeiChiXuShiJian2{ 0 };
        double yanChiTiFeiShiJian2{ 0 };
        double shangXianWei1{ 0 };
        double xiaXianWei1{ 0 };
        double zuoXianWei1{ 0 };
        double youXianWei1{ 0 };
        double xiangSuDangLiang1{ 0 };
        double shangXianWei2{ 0 };
        double xiaXianWei2{ 0 };
        double zuoXianWei2{ 0 };
        double youXianWei2{ 0 };
        double xiangSuDangLiang2{ 0 };
        double qiangBaoGuang{ 0 };
        double qiangZengYi{ 0 };
        double zhongBaoGuang{ 0 };
        double zhongZengYi{ 0 };
        double ruoBaoGuang{ 0 };
        double ruoZengYi{ 0 };
        bool saveNGImg{ false };
        bool saveMaskImg{ false };
        bool saveOKImg{ false };
        bool debugMode{ false };
		bool takeWork1Pictures{ false };
		bool takeWork2Pictures{ false };
    };

    inline SetConfig::SetConfig(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$SetConfig$")
        {
            throw std::runtime_error("Assembly is not $class$SetConfig$");
        }
        auto tiFeiChiXuShiJian1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tiFeiChiXuShiJian1$"));
        if (!tiFeiChiXuShiJian1Item) {
            throw std::runtime_error("$variable$tiFeiChiXuShiJian1 is not found");
        }
        tiFeiChiXuShiJian1 = tiFeiChiXuShiJian1Item->getValueAsDouble();
        auto yanChiTiFeiShiJian1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yanChiTiFeiShiJian1$"));
        if (!yanChiTiFeiShiJian1Item) {
            throw std::runtime_error("$variable$yanChiTiFeiShiJian1 is not found");
        }
        yanChiTiFeiShiJian1 = yanChiTiFeiShiJian1Item->getValueAsDouble();
        auto tiFeiChiXuShiJian2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tiFeiChiXuShiJian2$"));
        if (!tiFeiChiXuShiJian2Item) {
            throw std::runtime_error("$variable$tiFeiChiXuShiJian2 is not found");
        }
        tiFeiChiXuShiJian2 = tiFeiChiXuShiJian2Item->getValueAsDouble();
        auto yanChiTiFeiShiJian2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yanChiTiFeiShiJian2$"));
        if (!yanChiTiFeiShiJian2Item) {
            throw std::runtime_error("$variable$yanChiTiFeiShiJian2 is not found");
        }
        yanChiTiFeiShiJian2 = yanChiTiFeiShiJian2Item->getValueAsDouble();
        auto shangXianWei1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shangXianWei1$"));
        if (!shangXianWei1Item) {
            throw std::runtime_error("$variable$shangXianWei1 is not found");
        }
        shangXianWei1 = shangXianWei1Item->getValueAsDouble();
        auto xiaXianWei1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiaXianWei1$"));
        if (!xiaXianWei1Item) {
            throw std::runtime_error("$variable$xiaXianWei1 is not found");
        }
        xiaXianWei1 = xiaXianWei1Item->getValueAsDouble();
        auto zuoXianWei1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zuoXianWei1$"));
        if (!zuoXianWei1Item) {
            throw std::runtime_error("$variable$zuoXianWei1 is not found");
        }
        zuoXianWei1 = zuoXianWei1Item->getValueAsDouble();
        auto youXianWei1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$youXianWei1$"));
        if (!youXianWei1Item) {
            throw std::runtime_error("$variable$youXianWei1 is not found");
        }
        youXianWei1 = youXianWei1Item->getValueAsDouble();
        auto xiangSuDangLiang1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiangSuDangLiang1$"));
        if (!xiangSuDangLiang1Item) {
            throw std::runtime_error("$variable$xiangSuDangLiang1 is not found");
        }
        xiangSuDangLiang1 = xiangSuDangLiang1Item->getValueAsDouble();
        auto shangXianWei2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shangXianWei2$"));
        if (!shangXianWei2Item) {
            throw std::runtime_error("$variable$shangXianWei2 is not found");
        }
        shangXianWei2 = shangXianWei2Item->getValueAsDouble();
        auto xiaXianWei2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiaXianWei2$"));
        if (!xiaXianWei2Item) {
            throw std::runtime_error("$variable$xiaXianWei2 is not found");
        }
        xiaXianWei2 = xiaXianWei2Item->getValueAsDouble();
        auto zuoXianWei2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zuoXianWei2$"));
        if (!zuoXianWei2Item) {
            throw std::runtime_error("$variable$zuoXianWei2 is not found");
        }
        zuoXianWei2 = zuoXianWei2Item->getValueAsDouble();
        auto youXianWei2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$youXianWei2$"));
        if (!youXianWei2Item) {
            throw std::runtime_error("$variable$youXianWei2 is not found");
        }
        youXianWei2 = youXianWei2Item->getValueAsDouble();
        auto xiangSuDangLiang2Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiangSuDangLiang2$"));
        if (!xiangSuDangLiang2Item) {
            throw std::runtime_error("$variable$xiangSuDangLiang2 is not found");
        }
        xiangSuDangLiang2 = xiangSuDangLiang2Item->getValueAsDouble();
        auto qiangBaoGuangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiangBaoGuang$"));
        if (!qiangBaoGuangItem) {
            throw std::runtime_error("$variable$qiangBaoGuang is not found");
        }
        qiangBaoGuang = qiangBaoGuangItem->getValueAsDouble();
        auto qiangZengYiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiangZengYi$"));
        if (!qiangZengYiItem) {
            throw std::runtime_error("$variable$qiangZengYi is not found");
        }
        qiangZengYi = qiangZengYiItem->getValueAsDouble();
        auto zhongBaoGuangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zhongBaoGuang$"));
        if (!zhongBaoGuangItem) {
            throw std::runtime_error("$variable$zhongBaoGuang is not found");
        }
        zhongBaoGuang = zhongBaoGuangItem->getValueAsDouble();
        auto zhongZengYiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zhongZengYi$"));
        if (!zhongZengYiItem) {
            throw std::runtime_error("$variable$zhongZengYi is not found");
        }
        zhongZengYi = zhongZengYiItem->getValueAsDouble();
        auto ruoBaoGuangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$ruoBaoGuang$"));
        if (!ruoBaoGuangItem) {
            throw std::runtime_error("$variable$ruoBaoGuang is not found");
        }
        ruoBaoGuang = ruoBaoGuangItem->getValueAsDouble();
        auto ruoZengYiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$ruoZengYi$"));
        if (!ruoZengYiItem) {
            throw std::runtime_error("$variable$ruoZengYi is not found");
        }
        ruoZengYi = ruoZengYiItem->getValueAsDouble();
        auto saveNGImgItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$saveNGImg$"));
        if (!saveNGImgItem) {
            throw std::runtime_error("$variable$saveNGImg is not found");
        }
        saveNGImg = saveNGImgItem->getValueAsBool();
        auto saveMaskImgItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$saveMaskImg$"));
        if (!saveMaskImgItem) {
            throw std::runtime_error("$variable$saveMaskImg is not found");
        }
        saveMaskImg = saveMaskImgItem->getValueAsBool();
        auto saveOKImgItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$saveOKImg$"));
        if (!saveOKImgItem) {
            throw std::runtime_error("$variable$saveOKImg is not found");
        }
        saveOKImg = saveOKImgItem->getValueAsBool();
        auto debugModeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$debugMode$"));
        if (!debugModeItem) {
            throw std::runtime_error("$variable$debugMode is not found");
        }
        debugMode = debugModeItem->getValueAsBool();
    }

    inline SetConfig::SetConfig(const SetConfig& obj)
    {
        tiFeiChiXuShiJian1 = obj.tiFeiChiXuShiJian1;
        yanChiTiFeiShiJian1 = obj.yanChiTiFeiShiJian1;
        tiFeiChiXuShiJian2 = obj.tiFeiChiXuShiJian2;
        yanChiTiFeiShiJian2 = obj.yanChiTiFeiShiJian2;
        shangXianWei1 = obj.shangXianWei1;
        xiaXianWei1 = obj.xiaXianWei1;
        zuoXianWei1 = obj.zuoXianWei1;
        youXianWei1 = obj.youXianWei1;
        xiangSuDangLiang1 = obj.xiangSuDangLiang1;
        shangXianWei2 = obj.shangXianWei2;
        xiaXianWei2 = obj.xiaXianWei2;
        zuoXianWei2 = obj.zuoXianWei2;
        youXianWei2 = obj.youXianWei2;
        xiangSuDangLiang2 = obj.xiangSuDangLiang2;
        qiangBaoGuang = obj.qiangBaoGuang;
        qiangZengYi = obj.qiangZengYi;
        zhongBaoGuang = obj.zhongBaoGuang;
        zhongZengYi = obj.zhongZengYi;
        ruoBaoGuang = obj.ruoBaoGuang;
        ruoZengYi = obj.ruoZengYi;
        saveNGImg = obj.saveNGImg;
        saveMaskImg = obj.saveMaskImg;
        saveOKImg = obj.saveOKImg;
        debugMode = obj.debugMode;
    }

    inline SetConfig& SetConfig::operator=(const SetConfig& obj)
    {
        if (this != &obj) {
            tiFeiChiXuShiJian1 = obj.tiFeiChiXuShiJian1;
            yanChiTiFeiShiJian1 = obj.yanChiTiFeiShiJian1;
            tiFeiChiXuShiJian2 = obj.tiFeiChiXuShiJian2;
            yanChiTiFeiShiJian2 = obj.yanChiTiFeiShiJian2;
            shangXianWei1 = obj.shangXianWei1;
            xiaXianWei1 = obj.xiaXianWei1;
            zuoXianWei1 = obj.zuoXianWei1;
            youXianWei1 = obj.youXianWei1;
            xiangSuDangLiang1 = obj.xiangSuDangLiang1;
            shangXianWei2 = obj.shangXianWei2;
            xiaXianWei2 = obj.xiaXianWei2;
            zuoXianWei2 = obj.zuoXianWei2;
            youXianWei2 = obj.youXianWei2;
            xiangSuDangLiang2 = obj.xiangSuDangLiang2;
            qiangBaoGuang = obj.qiangBaoGuang;
            qiangZengYi = obj.qiangZengYi;
            zhongBaoGuang = obj.zhongBaoGuang;
            zhongZengYi = obj.zhongZengYi;
            ruoBaoGuang = obj.ruoBaoGuang;
            ruoZengYi = obj.ruoZengYi;
            saveNGImg = obj.saveNGImg;
            saveMaskImg = obj.saveMaskImg;
            saveOKImg = obj.saveOKImg;
            debugMode = obj.debugMode;
        }
        return *this;
    }

    inline SetConfig::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$SetConfig$");
        auto tiFeiChiXuShiJian1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        tiFeiChiXuShiJian1Item->setName("$variable$tiFeiChiXuShiJian1$");
        tiFeiChiXuShiJian1Item->setValueFromDouble(tiFeiChiXuShiJian1);
        assembly.addItem(tiFeiChiXuShiJian1Item);
        auto yanChiTiFeiShiJian1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        yanChiTiFeiShiJian1Item->setName("$variable$yanChiTiFeiShiJian1$");
        yanChiTiFeiShiJian1Item->setValueFromDouble(yanChiTiFeiShiJian1);
        assembly.addItem(yanChiTiFeiShiJian1Item);
        auto tiFeiChiXuShiJian2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        tiFeiChiXuShiJian2Item->setName("$variable$tiFeiChiXuShiJian2$");
        tiFeiChiXuShiJian2Item->setValueFromDouble(tiFeiChiXuShiJian2);
        assembly.addItem(tiFeiChiXuShiJian2Item);
        auto yanChiTiFeiShiJian2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        yanChiTiFeiShiJian2Item->setName("$variable$yanChiTiFeiShiJian2$");
        yanChiTiFeiShiJian2Item->setValueFromDouble(yanChiTiFeiShiJian2);
        assembly.addItem(yanChiTiFeiShiJian2Item);
        auto shangXianWei1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        shangXianWei1Item->setName("$variable$shangXianWei1$");
        shangXianWei1Item->setValueFromDouble(shangXianWei1);
        assembly.addItem(shangXianWei1Item);
        auto xiaXianWei1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        xiaXianWei1Item->setName("$variable$xiaXianWei1$");
        xiaXianWei1Item->setValueFromDouble(xiaXianWei1);
        assembly.addItem(xiaXianWei1Item);
        auto zuoXianWei1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        zuoXianWei1Item->setName("$variable$zuoXianWei1$");
        zuoXianWei1Item->setValueFromDouble(zuoXianWei1);
        assembly.addItem(zuoXianWei1Item);
        auto youXianWei1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        youXianWei1Item->setName("$variable$youXianWei1$");
        youXianWei1Item->setValueFromDouble(youXianWei1);
        assembly.addItem(youXianWei1Item);
        auto xiangSuDangLiang1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        xiangSuDangLiang1Item->setName("$variable$xiangSuDangLiang1$");
        xiangSuDangLiang1Item->setValueFromDouble(xiangSuDangLiang1);
        assembly.addItem(xiangSuDangLiang1Item);
        auto shangXianWei2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        shangXianWei2Item->setName("$variable$shangXianWei2$");
        shangXianWei2Item->setValueFromDouble(shangXianWei2);
        assembly.addItem(shangXianWei2Item);
        auto xiaXianWei2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        xiaXianWei2Item->setName("$variable$xiaXianWei2$");
        xiaXianWei2Item->setValueFromDouble(xiaXianWei2);
        assembly.addItem(xiaXianWei2Item);
        auto zuoXianWei2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        zuoXianWei2Item->setName("$variable$zuoXianWei2$");
        zuoXianWei2Item->setValueFromDouble(zuoXianWei2);
        assembly.addItem(zuoXianWei2Item);
        auto youXianWei2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        youXianWei2Item->setName("$variable$youXianWei2$");
        youXianWei2Item->setValueFromDouble(youXianWei2);
        assembly.addItem(youXianWei2Item);
        auto xiangSuDangLiang2Item = std::make_shared<rw::oso::ObjectStoreItem>();
        xiangSuDangLiang2Item->setName("$variable$xiangSuDangLiang2$");
        xiangSuDangLiang2Item->setValueFromDouble(xiangSuDangLiang2);
        assembly.addItem(xiangSuDangLiang2Item);
        auto qiangBaoGuangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiangBaoGuangItem->setName("$variable$qiangBaoGuang$");
        qiangBaoGuangItem->setValueFromDouble(qiangBaoGuang);
        assembly.addItem(qiangBaoGuangItem);
        auto qiangZengYiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiangZengYiItem->setName("$variable$qiangZengYi$");
        qiangZengYiItem->setValueFromDouble(qiangZengYi);
        assembly.addItem(qiangZengYiItem);
        auto zhongBaoGuangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zhongBaoGuangItem->setName("$variable$zhongBaoGuang$");
        zhongBaoGuangItem->setValueFromDouble(zhongBaoGuang);
        assembly.addItem(zhongBaoGuangItem);
        auto zhongZengYiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zhongZengYiItem->setName("$variable$zhongZengYi$");
        zhongZengYiItem->setValueFromDouble(zhongZengYi);
        assembly.addItem(zhongZengYiItem);
        auto ruoBaoGuangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        ruoBaoGuangItem->setName("$variable$ruoBaoGuang$");
        ruoBaoGuangItem->setValueFromDouble(ruoBaoGuang);
        assembly.addItem(ruoBaoGuangItem);
        auto ruoZengYiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        ruoZengYiItem->setName("$variable$ruoZengYi$");
        ruoZengYiItem->setValueFromDouble(ruoZengYi);
        assembly.addItem(ruoZengYiItem);
        auto saveNGImgItem = std::make_shared<rw::oso::ObjectStoreItem>();
        saveNGImgItem->setName("$variable$saveNGImg$");
        saveNGImgItem->setValueFromBool(saveNGImg);
        assembly.addItem(saveNGImgItem);
        auto saveMaskImgItem = std::make_shared<rw::oso::ObjectStoreItem>();
        saveMaskImgItem->setName("$variable$saveMaskImg$");
        saveMaskImgItem->setValueFromBool(saveMaskImg);
        assembly.addItem(saveMaskImgItem);
        auto saveOKImgItem = std::make_shared<rw::oso::ObjectStoreItem>();
        saveOKImgItem->setName("$variable$saveOKImg$");
        saveOKImgItem->setValueFromBool(saveOKImg);
        assembly.addItem(saveOKImgItem);
        auto debugModeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        debugModeItem->setName("$variable$debugMode$");
        debugModeItem->setValueFromBool(debugMode);
        assembly.addItem(debugModeItem);
        return assembly;
    }

    inline bool SetConfig::operator==(const SetConfig& obj) const
    {
        return tiFeiChiXuShiJian1 == obj.tiFeiChiXuShiJian1 && yanChiTiFeiShiJian1 == obj.yanChiTiFeiShiJian1 && tiFeiChiXuShiJian2 == obj.tiFeiChiXuShiJian2 && yanChiTiFeiShiJian2 == obj.yanChiTiFeiShiJian2 && shangXianWei1 == obj.shangXianWei1 && xiaXianWei1 == obj.xiaXianWei1 && zuoXianWei1 == obj.zuoXianWei1 && youXianWei1 == obj.youXianWei1 && xiangSuDangLiang1 == obj.xiangSuDangLiang1 && shangXianWei2 == obj.shangXianWei2 && xiaXianWei2 == obj.xiaXianWei2 && zuoXianWei2 == obj.zuoXianWei2 && youXianWei2 == obj.youXianWei2 && xiangSuDangLiang2 == obj.xiangSuDangLiang2 && qiangBaoGuang == obj.qiangBaoGuang && qiangZengYi == obj.qiangZengYi && zhongBaoGuang == obj.zhongBaoGuang && zhongZengYi == obj.zhongZengYi && ruoBaoGuang == obj.ruoBaoGuang && ruoZengYi == obj.ruoZengYi && saveNGImg == obj.saveNGImg && saveMaskImg == obj.saveMaskImg && saveOKImg == obj.saveOKImg && debugMode == obj.debugMode;
    }

    inline bool SetConfig::operator!=(const SetConfig& obj) const
    {
        return !(*this == obj);
    }

}

