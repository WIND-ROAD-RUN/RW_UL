#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class SetConfigSmartCroppingOfBags
    {
    public:
        SetConfigSmartCroppingOfBags() = default;
        ~SetConfigSmartCroppingOfBags() = default;

        SetConfigSmartCroppingOfBags(const rw::oso::ObjectStoreAssembly& assembly);
        SetConfigSmartCroppingOfBags(const SetConfigSmartCroppingOfBags& obj);

        SetConfigSmartCroppingOfBags& operator=(const SetConfigSmartCroppingOfBags& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const SetConfigSmartCroppingOfBags& obj) const;
        bool operator!=(const SetConfigSmartCroppingOfBags& obj) const;

    public:
        bool xiaopodong{ false };
        bool tiqiantifei{ false };
        bool xiangjitiaoshi{ false };
        bool qiyonger{ false };
        bool isxiangjizengyi{ false };
        double zidongpingbifanwei{ 0 };
        double pingjunmaichong{ 0 };
        double maichongxinhao{ 0 };
        double hanggao{ 0 };
        double daichang{ 0 };
        double daichangxishu{ 0 };
        double guasijuli{ 0 };
        double zuixiaodaichang{ 0 };
        double zuidadaichang{ 0 };
        double baisedailiangdufanweimin{ 0 };
        double baisedailiangdufanweimax{ 0 };
        double daokoudaoxiangjiluli{ 0 };
        double tifeiyanshi{ 0 };
        double baojingyanshi{ 0 };
        double tifeishijian{ 0 };
        double baojingshijian{ 0 };
        double chuiqiyanshi{ 0 };
        double dudaiyanshi{ 0 };
        double chuiqishijian{ 0 };
        double dudaishijian{ 0 };
        double maichongxishu{ 0 };
        double xiangjizengyi{ 0 };
        double houfenpinqi{ 0 };
        double chengfaqi{ 0 };
        double qiedaoxianshangpingbi{ 0 };
        double qiedaoxianxiapingbi{ 0 };
        double yansedailiangdufanweimin{ 0 };
        double yansedailiangdufanweimax{ 0 };
    };

    inline SetConfigSmartCroppingOfBags::SetConfigSmartCroppingOfBags(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$SetConfigSmartCroppingOfBags$")
        {
            throw std::runtime_error("Assembly is not $class$SetConfigSmartCroppingOfBags$");
        }
        auto xiaopodongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiaopodong$"));
        if (!xiaopodongItem) {
            throw std::runtime_error("$variable$xiaopodong is not found");
        }
        xiaopodong = xiaopodongItem->getValueAsBool();
        auto tiqiantifeiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tiqiantifei$"));
        if (!tiqiantifeiItem) {
            throw std::runtime_error("$variable$tiqiantifei is not found");
        }
        tiqiantifei = tiqiantifeiItem->getValueAsBool();
        auto xiangjitiaoshiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiangjitiaoshi$"));
        if (!xiangjitiaoshiItem) {
            throw std::runtime_error("$variable$xiangjitiaoshi is not found");
        }
        xiangjitiaoshi = xiangjitiaoshiItem->getValueAsBool();
        auto qiyongerItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiyonger$"));
        if (!qiyongerItem) {
            throw std::runtime_error("$variable$qiyonger is not found");
        }
        qiyonger = qiyongerItem->getValueAsBool();
        auto isxiangjizengyiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isxiangjizengyi$"));
        if (!isxiangjizengyiItem) {
            throw std::runtime_error("$variable$isxiangjizengyi is not found");
        }
        isxiangjizengyi = isxiangjizengyiItem->getValueAsBool();
        auto zidongpingbifanweiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zidongpingbifanwei$"));
        if (!zidongpingbifanweiItem) {
            throw std::runtime_error("$variable$zidongpingbifanwei is not found");
        }
        zidongpingbifanwei = zidongpingbifanweiItem->getValueAsDouble();
        auto pingjunmaichongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pingjunmaichong$"));
        if (!pingjunmaichongItem) {
            throw std::runtime_error("$variable$pingjunmaichong is not found");
        }
        pingjunmaichong = pingjunmaichongItem->getValueAsDouble();
        auto maichongxinhaoItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$maichongxinhao$"));
        if (!maichongxinhaoItem) {
            throw std::runtime_error("$variable$maichongxinhao is not found");
        }
        maichongxinhao = maichongxinhaoItem->getValueAsDouble();
        auto hanggaoItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$hanggao$"));
        if (!hanggaoItem) {
            throw std::runtime_error("$variable$hanggao is not found");
        }
        hanggao = hanggaoItem->getValueAsDouble();
        auto daichangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$daichang$"));
        if (!daichangItem) {
            throw std::runtime_error("$variable$daichang is not found");
        }
        daichang = daichangItem->getValueAsDouble();
        auto daichangxishuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$daichangxishu$"));
        if (!daichangxishuItem) {
            throw std::runtime_error("$variable$daichangxishu is not found");
        }
        daichangxishu = daichangxishuItem->getValueAsDouble();
        auto guasijuliItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$guasijuli$"));
        if (!guasijuliItem) {
            throw std::runtime_error("$variable$guasijuli is not found");
        }
        guasijuli = guasijuliItem->getValueAsDouble();
        auto zuixiaodaichangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zuixiaodaichang$"));
        if (!zuixiaodaichangItem) {
            throw std::runtime_error("$variable$zuixiaodaichang is not found");
        }
        zuixiaodaichang = zuixiaodaichangItem->getValueAsDouble();
        auto zuidadaichangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zuidadaichang$"));
        if (!zuidadaichangItem) {
            throw std::runtime_error("$variable$zuidadaichang is not found");
        }
        zuidadaichang = zuidadaichangItem->getValueAsDouble();
        auto baisedailiangdufanweiminItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baisedailiangdufanweimin$"));
        if (!baisedailiangdufanweiminItem) {
            throw std::runtime_error("$variable$baisedailiangdufanweimin is not found");
        }
        baisedailiangdufanweimin = baisedailiangdufanweiminItem->getValueAsDouble();
        auto baisedailiangdufanweimaxItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baisedailiangdufanweimax$"));
        if (!baisedailiangdufanweimaxItem) {
            throw std::runtime_error("$variable$baisedailiangdufanweimax is not found");
        }
        baisedailiangdufanweimax = baisedailiangdufanweimaxItem->getValueAsDouble();
        auto daokoudaoxiangjiluliItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$daokoudaoxiangjiluli$"));
        if (!daokoudaoxiangjiluliItem) {
            throw std::runtime_error("$variable$daokoudaoxiangjiluli is not found");
        }
        daokoudaoxiangjiluli = daokoudaoxiangjiluliItem->getValueAsDouble();
        auto tifeiyanshiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tifeiyanshi$"));
        if (!tifeiyanshiItem) {
            throw std::runtime_error("$variable$tifeiyanshi is not found");
        }
        tifeiyanshi = tifeiyanshiItem->getValueAsDouble();
        auto baojingyanshiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baojingyanshi$"));
        if (!baojingyanshiItem) {
            throw std::runtime_error("$variable$baojingyanshi is not found");
        }
        baojingyanshi = baojingyanshiItem->getValueAsDouble();
        auto tifeishijianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tifeishijian$"));
        if (!tifeishijianItem) {
            throw std::runtime_error("$variable$tifeishijian is not found");
        }
        tifeishijian = tifeishijianItem->getValueAsDouble();
        auto baojingshijianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baojingshijian$"));
        if (!baojingshijianItem) {
            throw std::runtime_error("$variable$baojingshijian is not found");
        }
        baojingshijian = baojingshijianItem->getValueAsDouble();
        auto chuiqiyanshiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chuiqiyanshi$"));
        if (!chuiqiyanshiItem) {
            throw std::runtime_error("$variable$chuiqiyanshi is not found");
        }
        chuiqiyanshi = chuiqiyanshiItem->getValueAsDouble();
        auto dudaiyanshiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$dudaiyanshi$"));
        if (!dudaiyanshiItem) {
            throw std::runtime_error("$variable$dudaiyanshi is not found");
        }
        dudaiyanshi = dudaiyanshiItem->getValueAsDouble();
        auto chuiqishijianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chuiqishijian$"));
        if (!chuiqishijianItem) {
            throw std::runtime_error("$variable$chuiqishijian is not found");
        }
        chuiqishijian = chuiqishijianItem->getValueAsDouble();
        auto dudaishijianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$dudaishijian$"));
        if (!dudaishijianItem) {
            throw std::runtime_error("$variable$dudaishijian is not found");
        }
        dudaishijian = dudaishijianItem->getValueAsDouble();
        auto maichongxishuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$maichongxishu$"));
        if (!maichongxishuItem) {
            throw std::runtime_error("$variable$maichongxishu is not found");
        }
        maichongxishu = maichongxishuItem->getValueAsDouble();
        auto xiangjizengyiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiangjizengyi$"));
        if (!xiangjizengyiItem) {
            throw std::runtime_error("$variable$xiangjizengyi is not found");
        }
        xiangjizengyi = xiangjizengyiItem->getValueAsDouble();
        auto houfenpinqiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$houfenpinqi$"));
        if (!houfenpinqiItem) {
            throw std::runtime_error("$variable$houfenpinqi is not found");
        }
        houfenpinqi = houfenpinqiItem->getValueAsDouble();
        auto chengfaqiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chengfaqi$"));
        if (!chengfaqiItem) {
            throw std::runtime_error("$variable$chengfaqi is not found");
        }
        chengfaqi = chengfaqiItem->getValueAsDouble();
        auto qiedaoxianshangpingbiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiedaoxianshangpingbi$"));
        if (!qiedaoxianshangpingbiItem) {
            throw std::runtime_error("$variable$qiedaoxianshangpingbi is not found");
        }
        qiedaoxianshangpingbi = qiedaoxianshangpingbiItem->getValueAsDouble();
        auto qiedaoxianxiapingbiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiedaoxianxiapingbi$"));
        if (!qiedaoxianxiapingbiItem) {
            throw std::runtime_error("$variable$qiedaoxianxiapingbi is not found");
        }
        qiedaoxianxiapingbi = qiedaoxianxiapingbiItem->getValueAsDouble();
        auto yansedailiangdufanweiminItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yansedailiangdufanweimin$"));
        if (!yansedailiangdufanweiminItem) {
            throw std::runtime_error("$variable$yansedailiangdufanweimin is not found");
        }
        yansedailiangdufanweimin = yansedailiangdufanweiminItem->getValueAsDouble();
        auto yansedailiangdufanweimaxItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yansedailiangdufanweimax$"));
        if (!yansedailiangdufanweimaxItem) {
            throw std::runtime_error("$variable$yansedailiangdufanweimax is not found");
        }
        yansedailiangdufanweimax = yansedailiangdufanweimaxItem->getValueAsDouble();
    }

    inline SetConfigSmartCroppingOfBags::SetConfigSmartCroppingOfBags(const SetConfigSmartCroppingOfBags& obj)
    {
        xiaopodong = obj.xiaopodong;
        tiqiantifei = obj.tiqiantifei;
        xiangjitiaoshi = obj.xiangjitiaoshi;
        qiyonger = obj.qiyonger;
        isxiangjizengyi = obj.isxiangjizengyi;
        zidongpingbifanwei = obj.zidongpingbifanwei;
        pingjunmaichong = obj.pingjunmaichong;
        maichongxinhao = obj.maichongxinhao;
        hanggao = obj.hanggao;
        daichang = obj.daichang;
        daichangxishu = obj.daichangxishu;
        guasijuli = obj.guasijuli;
        zuixiaodaichang = obj.zuixiaodaichang;
        zuidadaichang = obj.zuidadaichang;
        baisedailiangdufanweimin = obj.baisedailiangdufanweimin;
        baisedailiangdufanweimax = obj.baisedailiangdufanweimax;
        daokoudaoxiangjiluli = obj.daokoudaoxiangjiluli;
        tifeiyanshi = obj.tifeiyanshi;
        baojingyanshi = obj.baojingyanshi;
        tifeishijian = obj.tifeishijian;
        baojingshijian = obj.baojingshijian;
        chuiqiyanshi = obj.chuiqiyanshi;
        dudaiyanshi = obj.dudaiyanshi;
        chuiqishijian = obj.chuiqishijian;
        dudaishijian = obj.dudaishijian;
        maichongxishu = obj.maichongxishu;
        xiangjizengyi = obj.xiangjizengyi;
        houfenpinqi = obj.houfenpinqi;
        chengfaqi = obj.chengfaqi;
        qiedaoxianshangpingbi = obj.qiedaoxianshangpingbi;
        qiedaoxianxiapingbi = obj.qiedaoxianxiapingbi;
        yansedailiangdufanweimin = obj.yansedailiangdufanweimin;
        yansedailiangdufanweimax = obj.yansedailiangdufanweimax;
    }

    inline SetConfigSmartCroppingOfBags& SetConfigSmartCroppingOfBags::operator=(const SetConfigSmartCroppingOfBags& obj)
    {
        if (this != &obj) {
            xiaopodong = obj.xiaopodong;
            tiqiantifei = obj.tiqiantifei;
            xiangjitiaoshi = obj.xiangjitiaoshi;
            qiyonger = obj.qiyonger;
            isxiangjizengyi = obj.isxiangjizengyi;
            zidongpingbifanwei = obj.zidongpingbifanwei;
            pingjunmaichong = obj.pingjunmaichong;
            maichongxinhao = obj.maichongxinhao;
            hanggao = obj.hanggao;
            daichang = obj.daichang;
            daichangxishu = obj.daichangxishu;
            guasijuli = obj.guasijuli;
            zuixiaodaichang = obj.zuixiaodaichang;
            zuidadaichang = obj.zuidadaichang;
            baisedailiangdufanweimin = obj.baisedailiangdufanweimin;
            baisedailiangdufanweimax = obj.baisedailiangdufanweimax;
            daokoudaoxiangjiluli = obj.daokoudaoxiangjiluli;
            tifeiyanshi = obj.tifeiyanshi;
            baojingyanshi = obj.baojingyanshi;
            tifeishijian = obj.tifeishijian;
            baojingshijian = obj.baojingshijian;
            chuiqiyanshi = obj.chuiqiyanshi;
            dudaiyanshi = obj.dudaiyanshi;
            chuiqishijian = obj.chuiqishijian;
            dudaishijian = obj.dudaishijian;
            maichongxishu = obj.maichongxishu;
            xiangjizengyi = obj.xiangjizengyi;
            houfenpinqi = obj.houfenpinqi;
            chengfaqi = obj.chengfaqi;
            qiedaoxianshangpingbi = obj.qiedaoxianshangpingbi;
            qiedaoxianxiapingbi = obj.qiedaoxianxiapingbi;
            yansedailiangdufanweimin = obj.yansedailiangdufanweimin;
            yansedailiangdufanweimax = obj.yansedailiangdufanweimax;
        }
        return *this;
    }

    inline SetConfigSmartCroppingOfBags::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$SetConfigSmartCroppingOfBags$");
        auto xiaopodongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xiaopodongItem->setName("$variable$xiaopodong$");
        xiaopodongItem->setValueFromBool(xiaopodong);
        assembly.addItem(xiaopodongItem);
        auto tiqiantifeiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        tiqiantifeiItem->setName("$variable$tiqiantifei$");
        tiqiantifeiItem->setValueFromBool(tiqiantifei);
        assembly.addItem(tiqiantifeiItem);
        auto xiangjitiaoshiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xiangjitiaoshiItem->setName("$variable$xiangjitiaoshi$");
        xiangjitiaoshiItem->setValueFromBool(xiangjitiaoshi);
        assembly.addItem(xiangjitiaoshiItem);
        auto qiyongerItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiyongerItem->setName("$variable$qiyonger$");
        qiyongerItem->setValueFromBool(qiyonger);
        assembly.addItem(qiyongerItem);
        auto isxiangjizengyiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        isxiangjizengyiItem->setName("$variable$isxiangjizengyi$");
        isxiangjizengyiItem->setValueFromBool(isxiangjizengyi);
        assembly.addItem(isxiangjizengyiItem);
        auto zidongpingbifanweiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zidongpingbifanweiItem->setName("$variable$zidongpingbifanwei$");
        zidongpingbifanweiItem->setValueFromDouble(zidongpingbifanwei);
        assembly.addItem(zidongpingbifanweiItem);
        auto pingjunmaichongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        pingjunmaichongItem->setName("$variable$pingjunmaichong$");
        pingjunmaichongItem->setValueFromDouble(pingjunmaichong);
        assembly.addItem(pingjunmaichongItem);
        auto maichongxinhaoItem = std::make_shared<rw::oso::ObjectStoreItem>();
        maichongxinhaoItem->setName("$variable$maichongxinhao$");
        maichongxinhaoItem->setValueFromDouble(maichongxinhao);
        assembly.addItem(maichongxinhaoItem);
        auto hanggaoItem = std::make_shared<rw::oso::ObjectStoreItem>();
        hanggaoItem->setName("$variable$hanggao$");
        hanggaoItem->setValueFromDouble(hanggao);
        assembly.addItem(hanggaoItem);
        auto daichangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        daichangItem->setName("$variable$daichang$");
        daichangItem->setValueFromDouble(daichang);
        assembly.addItem(daichangItem);
        auto daichangxishuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        daichangxishuItem->setName("$variable$daichangxishu$");
        daichangxishuItem->setValueFromDouble(daichangxishu);
        assembly.addItem(daichangxishuItem);
        auto guasijuliItem = std::make_shared<rw::oso::ObjectStoreItem>();
        guasijuliItem->setName("$variable$guasijuli$");
        guasijuliItem->setValueFromDouble(guasijuli);
        assembly.addItem(guasijuliItem);
        auto zuixiaodaichangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zuixiaodaichangItem->setName("$variable$zuixiaodaichang$");
        zuixiaodaichangItem->setValueFromDouble(zuixiaodaichang);
        assembly.addItem(zuixiaodaichangItem);
        auto zuidadaichangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zuidadaichangItem->setName("$variable$zuidadaichang$");
        zuidadaichangItem->setValueFromDouble(zuidadaichang);
        assembly.addItem(zuidadaichangItem);
        auto baisedailiangdufanweiminItem = std::make_shared<rw::oso::ObjectStoreItem>();
        baisedailiangdufanweiminItem->setName("$variable$baisedailiangdufanweimin$");
        baisedailiangdufanweiminItem->setValueFromDouble(baisedailiangdufanweimin);
        assembly.addItem(baisedailiangdufanweiminItem);
        auto baisedailiangdufanweimaxItem = std::make_shared<rw::oso::ObjectStoreItem>();
        baisedailiangdufanweimaxItem->setName("$variable$baisedailiangdufanweimax$");
        baisedailiangdufanweimaxItem->setValueFromDouble(baisedailiangdufanweimax);
        assembly.addItem(baisedailiangdufanweimaxItem);
        auto daokoudaoxiangjiluliItem = std::make_shared<rw::oso::ObjectStoreItem>();
        daokoudaoxiangjiluliItem->setName("$variable$daokoudaoxiangjiluli$");
        daokoudaoxiangjiluliItem->setValueFromDouble(daokoudaoxiangjiluli);
        assembly.addItem(daokoudaoxiangjiluliItem);
        auto tifeiyanshiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        tifeiyanshiItem->setName("$variable$tifeiyanshi$");
        tifeiyanshiItem->setValueFromDouble(tifeiyanshi);
        assembly.addItem(tifeiyanshiItem);
        auto baojingyanshiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        baojingyanshiItem->setName("$variable$baojingyanshi$");
        baojingyanshiItem->setValueFromDouble(baojingyanshi);
        assembly.addItem(baojingyanshiItem);
        auto tifeishijianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        tifeishijianItem->setName("$variable$tifeishijian$");
        tifeishijianItem->setValueFromDouble(tifeishijian);
        assembly.addItem(tifeishijianItem);
        auto baojingshijianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        baojingshijianItem->setName("$variable$baojingshijian$");
        baojingshijianItem->setValueFromDouble(baojingshijian);
        assembly.addItem(baojingshijianItem);
        auto chuiqiyanshiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        chuiqiyanshiItem->setName("$variable$chuiqiyanshi$");
        chuiqiyanshiItem->setValueFromDouble(chuiqiyanshi);
        assembly.addItem(chuiqiyanshiItem);
        auto dudaiyanshiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        dudaiyanshiItem->setName("$variable$dudaiyanshi$");
        dudaiyanshiItem->setValueFromDouble(dudaiyanshi);
        assembly.addItem(dudaiyanshiItem);
        auto chuiqishijianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        chuiqishijianItem->setName("$variable$chuiqishijian$");
        chuiqishijianItem->setValueFromDouble(chuiqishijian);
        assembly.addItem(chuiqishijianItem);
        auto dudaishijianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        dudaishijianItem->setName("$variable$dudaishijian$");
        dudaishijianItem->setValueFromDouble(dudaishijian);
        assembly.addItem(dudaishijianItem);
        auto maichongxishuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        maichongxishuItem->setName("$variable$maichongxishu$");
        maichongxishuItem->setValueFromDouble(maichongxishu);
        assembly.addItem(maichongxishuItem);
        auto xiangjizengyiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xiangjizengyiItem->setName("$variable$xiangjizengyi$");
        xiangjizengyiItem->setValueFromDouble(xiangjizengyi);
        assembly.addItem(xiangjizengyiItem);
        auto houfenpinqiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        houfenpinqiItem->setName("$variable$houfenpinqi$");
        houfenpinqiItem->setValueFromDouble(houfenpinqi);
        assembly.addItem(houfenpinqiItem);
        auto chengfaqiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        chengfaqiItem->setName("$variable$chengfaqi$");
        chengfaqiItem->setValueFromDouble(chengfaqi);
        assembly.addItem(chengfaqiItem);
        auto qiedaoxianshangpingbiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiedaoxianshangpingbiItem->setName("$variable$qiedaoxianshangpingbi$");
        qiedaoxianshangpingbiItem->setValueFromDouble(qiedaoxianshangpingbi);
        assembly.addItem(qiedaoxianshangpingbiItem);
        auto qiedaoxianxiapingbiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiedaoxianxiapingbiItem->setName("$variable$qiedaoxianxiapingbi$");
        qiedaoxianxiapingbiItem->setValueFromDouble(qiedaoxianxiapingbi);
        assembly.addItem(qiedaoxianxiapingbiItem);
        auto yansedailiangdufanweiminItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yansedailiangdufanweiminItem->setName("$variable$yansedailiangdufanweimin$");
        yansedailiangdufanweiminItem->setValueFromDouble(yansedailiangdufanweimin);
        assembly.addItem(yansedailiangdufanweiminItem);
        auto yansedailiangdufanweimaxItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yansedailiangdufanweimaxItem->setName("$variable$yansedailiangdufanweimax$");
        yansedailiangdufanweimaxItem->setValueFromDouble(yansedailiangdufanweimax);
        assembly.addItem(yansedailiangdufanweimaxItem);
        return assembly;
    }

    inline bool SetConfigSmartCroppingOfBags::operator==(const SetConfigSmartCroppingOfBags& obj) const
    {
        return xiaopodong == obj.xiaopodong && tiqiantifei == obj.tiqiantifei && xiangjitiaoshi == obj.xiangjitiaoshi && qiyonger == obj.qiyonger && isxiangjizengyi == obj.isxiangjizengyi && zidongpingbifanwei == obj.zidongpingbifanwei && pingjunmaichong == obj.pingjunmaichong && maichongxinhao == obj.maichongxinhao && hanggao == obj.hanggao && daichang == obj.daichang && daichangxishu == obj.daichangxishu && guasijuli == obj.guasijuli && zuixiaodaichang == obj.zuixiaodaichang && zuidadaichang == obj.zuidadaichang && baisedailiangdufanweimin == obj.baisedailiangdufanweimin && baisedailiangdufanweimax == obj.baisedailiangdufanweimax && daokoudaoxiangjiluli == obj.daokoudaoxiangjiluli && tifeiyanshi == obj.tifeiyanshi && baojingyanshi == obj.baojingyanshi && tifeishijian == obj.tifeishijian && baojingshijian == obj.baojingshijian && chuiqiyanshi == obj.chuiqiyanshi && dudaiyanshi == obj.dudaiyanshi && chuiqishijian == obj.chuiqishijian && dudaishijian == obj.dudaishijian && maichongxishu == obj.maichongxishu && xiangjizengyi == obj.xiangjizengyi && houfenpinqi == obj.houfenpinqi && chengfaqi == obj.chengfaqi && qiedaoxianshangpingbi == obj.qiedaoxianshangpingbi && qiedaoxianxiapingbi == obj.qiedaoxianxiapingbi && yansedailiangdufanweimin == obj.yansedailiangdufanweimin && yansedailiangdufanweimax == obj.yansedailiangdufanweimax;
    }

    inline bool SetConfigSmartCroppingOfBags::operator!=(const SetConfigSmartCroppingOfBags& obj) const
    {
        return !(*this == obj);
    }

}

