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
        bool yundongkongzhiqichonglian{ false };
        double jiange{ 0 };
        double zidongpingbifanwei{ 0 };
        double pingjunmaichong1{ 0 };
        double maichongxinhao1{ 0 };
        double hanggao1{ 0 };
        double daichang1{ 0 };
        double daichangxishu1{ 0 };
        double guasijuli1{ 0 };
        double zuixiaodaichang1{ 0 };
        double zuidadaichang1{ 0 };
        double baisedailiangdufanweimin1{ 0 };
        double baisedailiangdufanweimax1{ 0 };
        double daokoudaoxiangjiluli1{ 0 };
        double xiangjibaoguang1{ 0 };
        double tifeiyanshi1{ 0 };
        double baojingyanshi1{ 0 };
        double tifeishijian1{ 0 };
        double baojingshijian1{ 0 };
        double chuiqiyanshi1{ 0 };
        double dudaiyanshi1{ 0 };
        double chuiqishijian1{ 0 };
        double dudaishijian1{ 0 };
        double maichongxishu1{ 0 };
        bool isxiangjizengyi1{ false };
        double xiangjizengyi1{ 0 };
        double houfenpinqi1{ 0 };
        double chengfaqi1{ 0 };
        double qiedaoxianshangpingbi1{ 0 };
        double qiedaoxianxiapingbi1{ 0 };
        double yansedailiangdufanweimin1{ 0 };
        double yansedailiangdufanweimax1{ 0 };
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
        auto yundongkongzhiqichonglianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yundongkongzhiqichonglian$"));
        if (!yundongkongzhiqichonglianItem) {
            throw std::runtime_error("$variable$yundongkongzhiqichonglian is not found");
        }
        yundongkongzhiqichonglian = yundongkongzhiqichonglianItem->getValueAsBool();
        auto jiangeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jiange$"));
        if (!jiangeItem) {
            throw std::runtime_error("$variable$jiange is not found");
        }
        jiange = jiangeItem->getValueAsDouble();
        auto zidongpingbifanweiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zidongpingbifanwei$"));
        if (!zidongpingbifanweiItem) {
            throw std::runtime_error("$variable$zidongpingbifanwei is not found");
        }
        zidongpingbifanwei = zidongpingbifanweiItem->getValueAsDouble();
        auto pingjunmaichong1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$pingjunmaichong1$"));
        if (!pingjunmaichong1Item) {
            throw std::runtime_error("$variable$pingjunmaichong1 is not found");
        }
        pingjunmaichong1 = pingjunmaichong1Item->getValueAsDouble();
        auto maichongxinhao1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$maichongxinhao1$"));
        if (!maichongxinhao1Item) {
            throw std::runtime_error("$variable$maichongxinhao1 is not found");
        }
        maichongxinhao1 = maichongxinhao1Item->getValueAsDouble();
        auto hanggao1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$hanggao1$"));
        if (!hanggao1Item) {
            throw std::runtime_error("$variable$hanggao1 is not found");
        }
        hanggao1 = hanggao1Item->getValueAsDouble();
        auto daichang1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$daichang1$"));
        if (!daichang1Item) {
            throw std::runtime_error("$variable$daichang1 is not found");
        }
        daichang1 = daichang1Item->getValueAsDouble();
        auto daichangxishu1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$daichangxishu1$"));
        if (!daichangxishu1Item) {
            throw std::runtime_error("$variable$daichangxishu1 is not found");
        }
        daichangxishu1 = daichangxishu1Item->getValueAsDouble();
        auto guasijuli1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$guasijuli1$"));
        if (!guasijuli1Item) {
            throw std::runtime_error("$variable$guasijuli1 is not found");
        }
        guasijuli1 = guasijuli1Item->getValueAsDouble();
        auto zuixiaodaichang1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zuixiaodaichang1$"));
        if (!zuixiaodaichang1Item) {
            throw std::runtime_error("$variable$zuixiaodaichang1 is not found");
        }
        zuixiaodaichang1 = zuixiaodaichang1Item->getValueAsDouble();
        auto zuidadaichang1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zuidadaichang1$"));
        if (!zuidadaichang1Item) {
            throw std::runtime_error("$variable$zuidadaichang1 is not found");
        }
        zuidadaichang1 = zuidadaichang1Item->getValueAsDouble();
        auto baisedailiangdufanweimin1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baisedailiangdufanweimin1$"));
        if (!baisedailiangdufanweimin1Item) {
            throw std::runtime_error("$variable$baisedailiangdufanweimin1 is not found");
        }
        baisedailiangdufanweimin1 = baisedailiangdufanweimin1Item->getValueAsDouble();
        auto baisedailiangdufanweimax1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baisedailiangdufanweimax1$"));
        if (!baisedailiangdufanweimax1Item) {
            throw std::runtime_error("$variable$baisedailiangdufanweimax1 is not found");
        }
        baisedailiangdufanweimax1 = baisedailiangdufanweimax1Item->getValueAsDouble();
        auto daokoudaoxiangjiluli1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$daokoudaoxiangjiluli1$"));
        if (!daokoudaoxiangjiluli1Item) {
            throw std::runtime_error("$variable$daokoudaoxiangjiluli1 is not found");
        }
        daokoudaoxiangjiluli1 = daokoudaoxiangjiluli1Item->getValueAsDouble();
        auto xiangjibaoguang1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiangjibaoguang1$"));
        if (!xiangjibaoguang1Item) {
            throw std::runtime_error("$variable$xiangjibaoguang1 is not found");
        }
        xiangjibaoguang1 = xiangjibaoguang1Item->getValueAsDouble();
        auto tifeiyanshi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tifeiyanshi1$"));
        if (!tifeiyanshi1Item) {
            throw std::runtime_error("$variable$tifeiyanshi1 is not found");
        }
        tifeiyanshi1 = tifeiyanshi1Item->getValueAsDouble();
        auto baojingyanshi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baojingyanshi1$"));
        if (!baojingyanshi1Item) {
            throw std::runtime_error("$variable$baojingyanshi1 is not found");
        }
        baojingyanshi1 = baojingyanshi1Item->getValueAsDouble();
        auto tifeishijian1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tifeishijian1$"));
        if (!tifeishijian1Item) {
            throw std::runtime_error("$variable$tifeishijian1 is not found");
        }
        tifeishijian1 = tifeishijian1Item->getValueAsDouble();
        auto baojingshijian1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$baojingshijian1$"));
        if (!baojingshijian1Item) {
            throw std::runtime_error("$variable$baojingshijian1 is not found");
        }
        baojingshijian1 = baojingshijian1Item->getValueAsDouble();
        auto chuiqiyanshi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chuiqiyanshi1$"));
        if (!chuiqiyanshi1Item) {
            throw std::runtime_error("$variable$chuiqiyanshi1 is not found");
        }
        chuiqiyanshi1 = chuiqiyanshi1Item->getValueAsDouble();
        auto dudaiyanshi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$dudaiyanshi1$"));
        if (!dudaiyanshi1Item) {
            throw std::runtime_error("$variable$dudaiyanshi1 is not found");
        }
        dudaiyanshi1 = dudaiyanshi1Item->getValueAsDouble();
        auto chuiqishijian1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chuiqishijian1$"));
        if (!chuiqishijian1Item) {
            throw std::runtime_error("$variable$chuiqishijian1 is not found");
        }
        chuiqishijian1 = chuiqishijian1Item->getValueAsDouble();
        auto dudaishijian1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$dudaishijian1$"));
        if (!dudaishijian1Item) {
            throw std::runtime_error("$variable$dudaishijian1 is not found");
        }
        dudaishijian1 = dudaishijian1Item->getValueAsDouble();
        auto maichongxishu1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$maichongxishu1$"));
        if (!maichongxishu1Item) {
            throw std::runtime_error("$variable$maichongxishu1 is not found");
        }
        maichongxishu1 = maichongxishu1Item->getValueAsDouble();
        auto isxiangjizengyi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$isxiangjizengyi1$"));
        if (!isxiangjizengyi1Item) {
            throw std::runtime_error("$variable$isxiangjizengyi1 is not found");
        }
        isxiangjizengyi1 = isxiangjizengyi1Item->getValueAsBool();
        auto xiangjizengyi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiangjizengyi1$"));
        if (!xiangjizengyi1Item) {
            throw std::runtime_error("$variable$xiangjizengyi1 is not found");
        }
        xiangjizengyi1 = xiangjizengyi1Item->getValueAsDouble();
        auto houfenpinqi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$houfenpinqi1$"));
        if (!houfenpinqi1Item) {
            throw std::runtime_error("$variable$houfenpinqi1 is not found");
        }
        houfenpinqi1 = houfenpinqi1Item->getValueAsDouble();
        auto chengfaqi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chengfaqi1$"));
        if (!chengfaqi1Item) {
            throw std::runtime_error("$variable$chengfaqi1 is not found");
        }
        chengfaqi1 = chengfaqi1Item->getValueAsDouble();
        auto qiedaoxianshangpingbi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiedaoxianshangpingbi1$"));
        if (!qiedaoxianshangpingbi1Item) {
            throw std::runtime_error("$variable$qiedaoxianshangpingbi1 is not found");
        }
        qiedaoxianshangpingbi1 = qiedaoxianshangpingbi1Item->getValueAsDouble();
        auto qiedaoxianxiapingbi1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiedaoxianxiapingbi1$"));
        if (!qiedaoxianxiapingbi1Item) {
            throw std::runtime_error("$variable$qiedaoxianxiapingbi1 is not found");
        }
        qiedaoxianxiapingbi1 = qiedaoxianxiapingbi1Item->getValueAsDouble();
        auto yansedailiangdufanweimin1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yansedailiangdufanweimin1$"));
        if (!yansedailiangdufanweimin1Item) {
            throw std::runtime_error("$variable$yansedailiangdufanweimin1 is not found");
        }
        yansedailiangdufanweimin1 = yansedailiangdufanweimin1Item->getValueAsDouble();
        auto yansedailiangdufanweimax1Item = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yansedailiangdufanweimax1$"));
        if (!yansedailiangdufanweimax1Item) {
            throw std::runtime_error("$variable$yansedailiangdufanweimax1 is not found");
        }
        yansedailiangdufanweimax1 = yansedailiangdufanweimax1Item->getValueAsDouble();
    }

    inline SetConfigSmartCroppingOfBags::SetConfigSmartCroppingOfBags(const SetConfigSmartCroppingOfBags& obj)
    {
        xiaopodong = obj.xiaopodong;
        tiqiantifei = obj.tiqiantifei;
        xiangjitiaoshi = obj.xiangjitiaoshi;
        qiyonger = obj.qiyonger;
        yundongkongzhiqichonglian = obj.yundongkongzhiqichonglian;
        jiange = obj.jiange;
        zidongpingbifanwei = obj.zidongpingbifanwei;
        pingjunmaichong1 = obj.pingjunmaichong1;
        maichongxinhao1 = obj.maichongxinhao1;
        hanggao1 = obj.hanggao1;
        daichang1 = obj.daichang1;
        daichangxishu1 = obj.daichangxishu1;
        guasijuli1 = obj.guasijuli1;
        zuixiaodaichang1 = obj.zuixiaodaichang1;
        zuidadaichang1 = obj.zuidadaichang1;
        baisedailiangdufanweimin1 = obj.baisedailiangdufanweimin1;
        baisedailiangdufanweimax1 = obj.baisedailiangdufanweimax1;
        daokoudaoxiangjiluli1 = obj.daokoudaoxiangjiluli1;
        xiangjibaoguang1 = obj.xiangjibaoguang1;
        tifeiyanshi1 = obj.tifeiyanshi1;
        baojingyanshi1 = obj.baojingyanshi1;
        tifeishijian1 = obj.tifeishijian1;
        baojingshijian1 = obj.baojingshijian1;
        chuiqiyanshi1 = obj.chuiqiyanshi1;
        dudaiyanshi1 = obj.dudaiyanshi1;
        chuiqishijian1 = obj.chuiqishijian1;
        dudaishijian1 = obj.dudaishijian1;
        maichongxishu1 = obj.maichongxishu1;
        isxiangjizengyi1 = obj.isxiangjizengyi1;
        xiangjizengyi1 = obj.xiangjizengyi1;
        houfenpinqi1 = obj.houfenpinqi1;
        chengfaqi1 = obj.chengfaqi1;
        qiedaoxianshangpingbi1 = obj.qiedaoxianshangpingbi1;
        qiedaoxianxiapingbi1 = obj.qiedaoxianxiapingbi1;
        yansedailiangdufanweimin1 = obj.yansedailiangdufanweimin1;
        yansedailiangdufanweimax1 = obj.yansedailiangdufanweimax1;
    }

    inline SetConfigSmartCroppingOfBags& SetConfigSmartCroppingOfBags::operator=(const SetConfigSmartCroppingOfBags& obj)
    {
        if (this != &obj) {
            xiaopodong = obj.xiaopodong;
            tiqiantifei = obj.tiqiantifei;
            xiangjitiaoshi = obj.xiangjitiaoshi;
            qiyonger = obj.qiyonger;
            yundongkongzhiqichonglian = obj.yundongkongzhiqichonglian;
            jiange = obj.jiange;
            zidongpingbifanwei = obj.zidongpingbifanwei;
            pingjunmaichong1 = obj.pingjunmaichong1;
            maichongxinhao1 = obj.maichongxinhao1;
            hanggao1 = obj.hanggao1;
            daichang1 = obj.daichang1;
            daichangxishu1 = obj.daichangxishu1;
            guasijuli1 = obj.guasijuli1;
            zuixiaodaichang1 = obj.zuixiaodaichang1;
            zuidadaichang1 = obj.zuidadaichang1;
            baisedailiangdufanweimin1 = obj.baisedailiangdufanweimin1;
            baisedailiangdufanweimax1 = obj.baisedailiangdufanweimax1;
            daokoudaoxiangjiluli1 = obj.daokoudaoxiangjiluli1;
            xiangjibaoguang1 = obj.xiangjibaoguang1;
            tifeiyanshi1 = obj.tifeiyanshi1;
            baojingyanshi1 = obj.baojingyanshi1;
            tifeishijian1 = obj.tifeishijian1;
            baojingshijian1 = obj.baojingshijian1;
            chuiqiyanshi1 = obj.chuiqiyanshi1;
            dudaiyanshi1 = obj.dudaiyanshi1;
            chuiqishijian1 = obj.chuiqishijian1;
            dudaishijian1 = obj.dudaishijian1;
            maichongxishu1 = obj.maichongxishu1;
            isxiangjizengyi1 = obj.isxiangjizengyi1;
            xiangjizengyi1 = obj.xiangjizengyi1;
            houfenpinqi1 = obj.houfenpinqi1;
            chengfaqi1 = obj.chengfaqi1;
            qiedaoxianshangpingbi1 = obj.qiedaoxianshangpingbi1;
            qiedaoxianxiapingbi1 = obj.qiedaoxianxiapingbi1;
            yansedailiangdufanweimin1 = obj.yansedailiangdufanweimin1;
            yansedailiangdufanweimax1 = obj.yansedailiangdufanweimax1;
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
        auto yundongkongzhiqichonglianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yundongkongzhiqichonglianItem->setName("$variable$yundongkongzhiqichonglian$");
        yundongkongzhiqichonglianItem->setValueFromBool(yundongkongzhiqichonglian);
        assembly.addItem(yundongkongzhiqichonglianItem);
        auto jiangeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jiangeItem->setName("$variable$jiange$");
        jiangeItem->setValueFromDouble(jiange);
        assembly.addItem(jiangeItem);
        auto zidongpingbifanweiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zidongpingbifanweiItem->setName("$variable$zidongpingbifanwei$");
        zidongpingbifanweiItem->setValueFromDouble(zidongpingbifanwei);
        assembly.addItem(zidongpingbifanweiItem);
        auto pingjunmaichong1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        pingjunmaichong1Item->setName("$variable$pingjunmaichong1$");
        pingjunmaichong1Item->setValueFromDouble(pingjunmaichong1);
        assembly.addItem(pingjunmaichong1Item);
        auto maichongxinhao1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        maichongxinhao1Item->setName("$variable$maichongxinhao1$");
        maichongxinhao1Item->setValueFromDouble(maichongxinhao1);
        assembly.addItem(maichongxinhao1Item);
        auto hanggao1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        hanggao1Item->setName("$variable$hanggao1$");
        hanggao1Item->setValueFromDouble(hanggao1);
        assembly.addItem(hanggao1Item);
        auto daichang1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        daichang1Item->setName("$variable$daichang1$");
        daichang1Item->setValueFromDouble(daichang1);
        assembly.addItem(daichang1Item);
        auto daichangxishu1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        daichangxishu1Item->setName("$variable$daichangxishu1$");
        daichangxishu1Item->setValueFromDouble(daichangxishu1);
        assembly.addItem(daichangxishu1Item);
        auto guasijuli1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        guasijuli1Item->setName("$variable$guasijuli1$");
        guasijuli1Item->setValueFromDouble(guasijuli1);
        assembly.addItem(guasijuli1Item);
        auto zuixiaodaichang1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        zuixiaodaichang1Item->setName("$variable$zuixiaodaichang1$");
        zuixiaodaichang1Item->setValueFromDouble(zuixiaodaichang1);
        assembly.addItem(zuixiaodaichang1Item);
        auto zuidadaichang1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        zuidadaichang1Item->setName("$variable$zuidadaichang1$");
        zuidadaichang1Item->setValueFromDouble(zuidadaichang1);
        assembly.addItem(zuidadaichang1Item);
        auto baisedailiangdufanweimin1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        baisedailiangdufanweimin1Item->setName("$variable$baisedailiangdufanweimin1$");
        baisedailiangdufanweimin1Item->setValueFromDouble(baisedailiangdufanweimin1);
        assembly.addItem(baisedailiangdufanweimin1Item);
        auto baisedailiangdufanweimax1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        baisedailiangdufanweimax1Item->setName("$variable$baisedailiangdufanweimax1$");
        baisedailiangdufanweimax1Item->setValueFromDouble(baisedailiangdufanweimax1);
        assembly.addItem(baisedailiangdufanweimax1Item);
        auto daokoudaoxiangjiluli1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        daokoudaoxiangjiluli1Item->setName("$variable$daokoudaoxiangjiluli1$");
        daokoudaoxiangjiluli1Item->setValueFromDouble(daokoudaoxiangjiluli1);
        assembly.addItem(daokoudaoxiangjiluli1Item);
        auto xiangjibaoguang1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        xiangjibaoguang1Item->setName("$variable$xiangjibaoguang1$");
        xiangjibaoguang1Item->setValueFromDouble(xiangjibaoguang1);
        assembly.addItem(xiangjibaoguang1Item);
        auto tifeiyanshi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        tifeiyanshi1Item->setName("$variable$tifeiyanshi1$");
        tifeiyanshi1Item->setValueFromDouble(tifeiyanshi1);
        assembly.addItem(tifeiyanshi1Item);
        auto baojingyanshi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        baojingyanshi1Item->setName("$variable$baojingyanshi1$");
        baojingyanshi1Item->setValueFromDouble(baojingyanshi1);
        assembly.addItem(baojingyanshi1Item);
        auto tifeishijian1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        tifeishijian1Item->setName("$variable$tifeishijian1$");
        tifeishijian1Item->setValueFromDouble(tifeishijian1);
        assembly.addItem(tifeishijian1Item);
        auto baojingshijian1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        baojingshijian1Item->setName("$variable$baojingshijian1$");
        baojingshijian1Item->setValueFromDouble(baojingshijian1);
        assembly.addItem(baojingshijian1Item);
        auto chuiqiyanshi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        chuiqiyanshi1Item->setName("$variable$chuiqiyanshi1$");
        chuiqiyanshi1Item->setValueFromDouble(chuiqiyanshi1);
        assembly.addItem(chuiqiyanshi1Item);
        auto dudaiyanshi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        dudaiyanshi1Item->setName("$variable$dudaiyanshi1$");
        dudaiyanshi1Item->setValueFromDouble(dudaiyanshi1);
        assembly.addItem(dudaiyanshi1Item);
        auto chuiqishijian1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        chuiqishijian1Item->setName("$variable$chuiqishijian1$");
        chuiqishijian1Item->setValueFromDouble(chuiqishijian1);
        assembly.addItem(chuiqishijian1Item);
        auto dudaishijian1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        dudaishijian1Item->setName("$variable$dudaishijian1$");
        dudaishijian1Item->setValueFromDouble(dudaishijian1);
        assembly.addItem(dudaishijian1Item);
        auto maichongxishu1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        maichongxishu1Item->setName("$variable$maichongxishu1$");
        maichongxishu1Item->setValueFromDouble(maichongxishu1);
        assembly.addItem(maichongxishu1Item);
        auto isxiangjizengyi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        isxiangjizengyi1Item->setName("$variable$isxiangjizengyi1$");
        isxiangjizengyi1Item->setValueFromBool(isxiangjizengyi1);
        assembly.addItem(isxiangjizengyi1Item);
        auto xiangjizengyi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        xiangjizengyi1Item->setName("$variable$xiangjizengyi1$");
        xiangjizengyi1Item->setValueFromDouble(xiangjizengyi1);
        assembly.addItem(xiangjizengyi1Item);
        auto houfenpinqi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        houfenpinqi1Item->setName("$variable$houfenpinqi1$");
        houfenpinqi1Item->setValueFromDouble(houfenpinqi1);
        assembly.addItem(houfenpinqi1Item);
        auto chengfaqi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        chengfaqi1Item->setName("$variable$chengfaqi1$");
        chengfaqi1Item->setValueFromDouble(chengfaqi1);
        assembly.addItem(chengfaqi1Item);
        auto qiedaoxianshangpingbi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        qiedaoxianshangpingbi1Item->setName("$variable$qiedaoxianshangpingbi1$");
        qiedaoxianshangpingbi1Item->setValueFromDouble(qiedaoxianshangpingbi1);
        assembly.addItem(qiedaoxianshangpingbi1Item);
        auto qiedaoxianxiapingbi1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        qiedaoxianxiapingbi1Item->setName("$variable$qiedaoxianxiapingbi1$");
        qiedaoxianxiapingbi1Item->setValueFromDouble(qiedaoxianxiapingbi1);
        assembly.addItem(qiedaoxianxiapingbi1Item);
        auto yansedailiangdufanweimin1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        yansedailiangdufanweimin1Item->setName("$variable$yansedailiangdufanweimin1$");
        yansedailiangdufanweimin1Item->setValueFromDouble(yansedailiangdufanweimin1);
        assembly.addItem(yansedailiangdufanweimin1Item);
        auto yansedailiangdufanweimax1Item = std::make_shared<rw::oso::ObjectStoreItem>();
        yansedailiangdufanweimax1Item->setName("$variable$yansedailiangdufanweimax1$");
        yansedailiangdufanweimax1Item->setValueFromDouble(yansedailiangdufanweimax1);
        assembly.addItem(yansedailiangdufanweimax1Item);
        return assembly;
    }

    inline bool SetConfigSmartCroppingOfBags::operator==(const SetConfigSmartCroppingOfBags& obj) const
    {
        return xiaopodong == obj.xiaopodong && tiqiantifei == obj.tiqiantifei && xiangjitiaoshi == obj.xiangjitiaoshi && qiyonger == obj.qiyonger && yundongkongzhiqichonglian == obj.yundongkongzhiqichonglian && jiange == obj.jiange && zidongpingbifanwei == obj.zidongpingbifanwei && pingjunmaichong1 == obj.pingjunmaichong1 && maichongxinhao1 == obj.maichongxinhao1 && hanggao1 == obj.hanggao1 && daichang1 == obj.daichang1 && daichangxishu1 == obj.daichangxishu1 && guasijuli1 == obj.guasijuli1 && zuixiaodaichang1 == obj.zuixiaodaichang1 && zuidadaichang1 == obj.zuidadaichang1 && baisedailiangdufanweimin1 == obj.baisedailiangdufanweimin1 && baisedailiangdufanweimax1 == obj.baisedailiangdufanweimax1 && daokoudaoxiangjiluli1 == obj.daokoudaoxiangjiluli1 && xiangjibaoguang1 == obj.xiangjibaoguang1 && tifeiyanshi1 == obj.tifeiyanshi1 && baojingyanshi1 == obj.baojingyanshi1 && tifeishijian1 == obj.tifeishijian1 && baojingshijian1 == obj.baojingshijian1 && chuiqiyanshi1 == obj.chuiqiyanshi1 && dudaiyanshi1 == obj.dudaiyanshi1 && chuiqishijian1 == obj.chuiqishijian1 && dudaishijian1 == obj.dudaishijian1 && maichongxishu1 == obj.maichongxishu1 && isxiangjizengyi1 == obj.isxiangjizengyi1 && xiangjizengyi1 == obj.xiangjizengyi1 && houfenpinqi1 == obj.houfenpinqi1 && chengfaqi1 == obj.chengfaqi1 && qiedaoxianshangpingbi1 == obj.qiedaoxianshangpingbi1 && qiedaoxianxiapingbi1 == obj.qiedaoxianxiapingbi1 && yansedailiangdufanweimin1 == obj.yansedailiangdufanweimin1 && yansedailiangdufanweimax1 == obj.yansedailiangdufanweimax1;
    }

    inline bool SetConfigSmartCroppingOfBags::operator!=(const SetConfigSmartCroppingOfBags& obj) const
    {
        return !(*this == obj);
    }

}

