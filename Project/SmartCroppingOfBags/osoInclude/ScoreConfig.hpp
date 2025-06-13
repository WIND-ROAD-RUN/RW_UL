#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ScoreConfigSmartCroppingOfBags
    {
    public:
        ScoreConfigSmartCroppingOfBags() = default;
        ~ScoreConfigSmartCroppingOfBags() = default;

        ScoreConfigSmartCroppingOfBags(const rw::oso::ObjectStoreAssembly& assembly);
        ScoreConfigSmartCroppingOfBags(const ScoreConfigSmartCroppingOfBags& obj);

        ScoreConfigSmartCroppingOfBags& operator=(const ScoreConfigSmartCroppingOfBags& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ScoreConfigSmartCroppingOfBags& obj) const;
        bool operator!=(const ScoreConfigSmartCroppingOfBags& obj) const;

    public:
        bool heiba{ false };
        double heibascore{ 0 };
        double heibaarea{ 0 };
        bool shudang{ false };
        double shudangscore{ 0 };
        double shudangarea{ 0 };
        bool huapo{ false };
        double huaposcore{ 0 };
        double huapoarea{ 0 };
        bool jietou{ false };
        double jietouscore{ 0 };
        double jietouarea{ 0 };
        bool guasi{ false };
        double guasiscore{ 0 };
        double guasiarea{ 0 };
        bool podong{ false };
        double podongscore{ 0 };
        double podongarea{ 0 };
        bool zangwu{ false };
        double zangwuscore{ 0 };
        double zangwuarea{ 0 };
        bool noshudang{ false };
        double noshudangscore{ 0 };
        double noshudangarea{ 0 };
        bool modian{ false };
        double modianscore{ 0 };
        double modianarea{ 0 };
        bool loumo{ false };
        double loumoscore{ 0 };
        double loumoarea{ 0 };
        bool xishudang{ false };
        double xishudangscore{ 0 };
        double xishudangarea{ 0 };
        bool erweima{ false };
        double erweimascore{ 0 };
        double erweimaarea{ 0 };
        bool damodian{ false };
        double damodianscore{ 0 };
        double damodianarea{ 0 };
        bool kongdong{ false };
        double kongdongscore{ 0 };
        double kongdongarea{ 0 };
        bool sebiao{ false };
        double sebiaoscore{ 0 };
        double sebiaoarea{ 0 };
        bool yinshuaquexian{ false };
        double yinshuaquexianscore{ 0 };
        double yinshuaquexianarea{ 0 };
        bool xiaopodong{ false };
        double xiaopodongscore{ 0 };
        double xiaopodongarea{ 0 };
        bool jiaodai{ false };
        double jiaodaiscore{ 0 };
        double jiaodaiarea{ 0 };
    };

    inline ScoreConfigSmartCroppingOfBags::ScoreConfigSmartCroppingOfBags(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ScoreConfigSmartCroppingOfBags$")
        {
            throw std::runtime_error("Assembly is not $class$ScoreConfigSmartCroppingOfBags$");
        }
        auto heibaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$heiba$"));
        if (!heibaItem) {
            throw std::runtime_error("$variable$heiba is not found");
        }
        heiba = heibaItem->getValueAsBool();
        auto heibascoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$heibascore$"));
        if (!heibascoreItem) {
            throw std::runtime_error("$variable$heibascore is not found");
        }
        heibascore = heibascoreItem->getValueAsDouble();
        auto heibaareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$heibaarea$"));
        if (!heibaareaItem) {
            throw std::runtime_error("$variable$heibaarea is not found");
        }
        heibaarea = heibaareaItem->getValueAsDouble();
        auto shudangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shudang$"));
        if (!shudangItem) {
            throw std::runtime_error("$variable$shudang is not found");
        }
        shudang = shudangItem->getValueAsBool();
        auto shudangscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shudangscore$"));
        if (!shudangscoreItem) {
            throw std::runtime_error("$variable$shudangscore is not found");
        }
        shudangscore = shudangscoreItem->getValueAsDouble();
        auto shudangareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shudangarea$"));
        if (!shudangareaItem) {
            throw std::runtime_error("$variable$shudangarea is not found");
        }
        shudangarea = shudangareaItem->getValueAsDouble();
        auto huapoItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$huapo$"));
        if (!huapoItem) {
            throw std::runtime_error("$variable$huapo is not found");
        }
        huapo = huapoItem->getValueAsBool();
        auto huaposcoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$huaposcore$"));
        if (!huaposcoreItem) {
            throw std::runtime_error("$variable$huaposcore is not found");
        }
        huaposcore = huaposcoreItem->getValueAsDouble();
        auto huapoareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$huapoarea$"));
        if (!huapoareaItem) {
            throw std::runtime_error("$variable$huapoarea is not found");
        }
        huapoarea = huapoareaItem->getValueAsDouble();
        auto jietouItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jietou$"));
        if (!jietouItem) {
            throw std::runtime_error("$variable$jietou is not found");
        }
        jietou = jietouItem->getValueAsBool();
        auto jietouscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jietouscore$"));
        if (!jietouscoreItem) {
            throw std::runtime_error("$variable$jietouscore is not found");
        }
        jietouscore = jietouscoreItem->getValueAsDouble();
        auto jietouareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jietouarea$"));
        if (!jietouareaItem) {
            throw std::runtime_error("$variable$jietouarea is not found");
        }
        jietouarea = jietouareaItem->getValueAsDouble();
        auto guasiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$guasi$"));
        if (!guasiItem) {
            throw std::runtime_error("$variable$guasi is not found");
        }
        guasi = guasiItem->getValueAsBool();
        auto guasiscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$guasiscore$"));
        if (!guasiscoreItem) {
            throw std::runtime_error("$variable$guasiscore is not found");
        }
        guasiscore = guasiscoreItem->getValueAsDouble();
        auto guasiareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$guasiarea$"));
        if (!guasiareaItem) {
            throw std::runtime_error("$variable$guasiarea is not found");
        }
        guasiarea = guasiareaItem->getValueAsDouble();
        auto podongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$podong$"));
        if (!podongItem) {
            throw std::runtime_error("$variable$podong is not found");
        }
        podong = podongItem->getValueAsBool();
        auto podongscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$podongscore$"));
        if (!podongscoreItem) {
            throw std::runtime_error("$variable$podongscore is not found");
        }
        podongscore = podongscoreItem->getValueAsDouble();
        auto podongareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$podongarea$"));
        if (!podongareaItem) {
            throw std::runtime_error("$variable$podongarea is not found");
        }
        podongarea = podongareaItem->getValueAsDouble();
        auto zangwuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zangwu$"));
        if (!zangwuItem) {
            throw std::runtime_error("$variable$zangwu is not found");
        }
        zangwu = zangwuItem->getValueAsBool();
        auto zangwuscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zangwuscore$"));
        if (!zangwuscoreItem) {
            throw std::runtime_error("$variable$zangwuscore is not found");
        }
        zangwuscore = zangwuscoreItem->getValueAsDouble();
        auto zangwuareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zangwuarea$"));
        if (!zangwuareaItem) {
            throw std::runtime_error("$variable$zangwuarea is not found");
        }
        zangwuarea = zangwuareaItem->getValueAsDouble();
        auto noshudangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$noshudang$"));
        if (!noshudangItem) {
            throw std::runtime_error("$variable$noshudang is not found");
        }
        noshudang = noshudangItem->getValueAsBool();
        auto noshudangscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$noshudangscore$"));
        if (!noshudangscoreItem) {
            throw std::runtime_error("$variable$noshudangscore is not found");
        }
        noshudangscore = noshudangscoreItem->getValueAsDouble();
        auto noshudangareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$noshudangarea$"));
        if (!noshudangareaItem) {
            throw std::runtime_error("$variable$noshudangarea is not found");
        }
        noshudangarea = noshudangareaItem->getValueAsDouble();
        auto modianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$modian$"));
        if (!modianItem) {
            throw std::runtime_error("$variable$modian is not found");
        }
        modian = modianItem->getValueAsBool();
        auto modianscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$modianscore$"));
        if (!modianscoreItem) {
            throw std::runtime_error("$variable$modianscore is not found");
        }
        modianscore = modianscoreItem->getValueAsDouble();
        auto modianareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$modianarea$"));
        if (!modianareaItem) {
            throw std::runtime_error("$variable$modianarea is not found");
        }
        modianarea = modianareaItem->getValueAsDouble();
        auto loumoItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$loumo$"));
        if (!loumoItem) {
            throw std::runtime_error("$variable$loumo is not found");
        }
        loumo = loumoItem->getValueAsBool();
        auto loumoscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$loumoscore$"));
        if (!loumoscoreItem) {
            throw std::runtime_error("$variable$loumoscore is not found");
        }
        loumoscore = loumoscoreItem->getValueAsDouble();
        auto loumoareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$loumoarea$"));
        if (!loumoareaItem) {
            throw std::runtime_error("$variable$loumoarea is not found");
        }
        loumoarea = loumoareaItem->getValueAsDouble();
        auto xishudangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xishudang$"));
        if (!xishudangItem) {
            throw std::runtime_error("$variable$xishudang is not found");
        }
        xishudang = xishudangItem->getValueAsBool();
        auto xishudangscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xishudangscore$"));
        if (!xishudangscoreItem) {
            throw std::runtime_error("$variable$xishudangscore is not found");
        }
        xishudangscore = xishudangscoreItem->getValueAsDouble();
        auto xishudangareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xishudangarea$"));
        if (!xishudangareaItem) {
            throw std::runtime_error("$variable$xishudangarea is not found");
        }
        xishudangarea = xishudangareaItem->getValueAsDouble();
        auto erweimaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$erweima$"));
        if (!erweimaItem) {
            throw std::runtime_error("$variable$erweima is not found");
        }
        erweima = erweimaItem->getValueAsBool();
        auto erweimascoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$erweimascore$"));
        if (!erweimascoreItem) {
            throw std::runtime_error("$variable$erweimascore is not found");
        }
        erweimascore = erweimascoreItem->getValueAsDouble();
        auto erweimaareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$erweimaarea$"));
        if (!erweimaareaItem) {
            throw std::runtime_error("$variable$erweimaarea is not found");
        }
        erweimaarea = erweimaareaItem->getValueAsDouble();
        auto damodianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$damodian$"));
        if (!damodianItem) {
            throw std::runtime_error("$variable$damodian is not found");
        }
        damodian = damodianItem->getValueAsBool();
        auto damodianscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$damodianscore$"));
        if (!damodianscoreItem) {
            throw std::runtime_error("$variable$damodianscore is not found");
        }
        damodianscore = damodianscoreItem->getValueAsDouble();
        auto damodianareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$damodianarea$"));
        if (!damodianareaItem) {
            throw std::runtime_error("$variable$damodianarea is not found");
        }
        damodianarea = damodianareaItem->getValueAsDouble();
        auto kongdongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$kongdong$"));
        if (!kongdongItem) {
            throw std::runtime_error("$variable$kongdong is not found");
        }
        kongdong = kongdongItem->getValueAsBool();
        auto kongdongscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$kongdongscore$"));
        if (!kongdongscoreItem) {
            throw std::runtime_error("$variable$kongdongscore is not found");
        }
        kongdongscore = kongdongscoreItem->getValueAsDouble();
        auto kongdongareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$kongdongarea$"));
        if (!kongdongareaItem) {
            throw std::runtime_error("$variable$kongdongarea is not found");
        }
        kongdongarea = kongdongareaItem->getValueAsDouble();
        auto sebiaoItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$sebiao$"));
        if (!sebiaoItem) {
            throw std::runtime_error("$variable$sebiao is not found");
        }
        sebiao = sebiaoItem->getValueAsBool();
        auto sebiaoscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$sebiaoscore$"));
        if (!sebiaoscoreItem) {
            throw std::runtime_error("$variable$sebiaoscore is not found");
        }
        sebiaoscore = sebiaoscoreItem->getValueAsDouble();
        auto sebiaoareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$sebiaoarea$"));
        if (!sebiaoareaItem) {
            throw std::runtime_error("$variable$sebiaoarea is not found");
        }
        sebiaoarea = sebiaoareaItem->getValueAsDouble();
        auto yinshuaquexianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yinshuaquexian$"));
        if (!yinshuaquexianItem) {
            throw std::runtime_error("$variable$yinshuaquexian is not found");
        }
        yinshuaquexian = yinshuaquexianItem->getValueAsBool();
        auto yinshuaquexianscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yinshuaquexianscore$"));
        if (!yinshuaquexianscoreItem) {
            throw std::runtime_error("$variable$yinshuaquexianscore is not found");
        }
        yinshuaquexianscore = yinshuaquexianscoreItem->getValueAsDouble();
        auto yinshuaquexianareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yinshuaquexianarea$"));
        if (!yinshuaquexianareaItem) {
            throw std::runtime_error("$variable$yinshuaquexianarea is not found");
        }
        yinshuaquexianarea = yinshuaquexianareaItem->getValueAsDouble();
        auto xiaopodongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiaopodong$"));
        if (!xiaopodongItem) {
            throw std::runtime_error("$variable$xiaopodong is not found");
        }
        xiaopodong = xiaopodongItem->getValueAsBool();
        auto xiaopodongscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiaopodongscore$"));
        if (!xiaopodongscoreItem) {
            throw std::runtime_error("$variable$xiaopodongscore is not found");
        }
        xiaopodongscore = xiaopodongscoreItem->getValueAsDouble();
        auto xiaopodongareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiaopodongarea$"));
        if (!xiaopodongareaItem) {
            throw std::runtime_error("$variable$xiaopodongarea is not found");
        }
        xiaopodongarea = xiaopodongareaItem->getValueAsDouble();
        auto jiaodaiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jiaodai$"));
        if (!jiaodaiItem) {
            throw std::runtime_error("$variable$jiaodai is not found");
        }
        jiaodai = jiaodaiItem->getValueAsBool();
        auto jiaodaiscoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jiaodaiscore$"));
        if (!jiaodaiscoreItem) {
            throw std::runtime_error("$variable$jiaodaiscore is not found");
        }
        jiaodaiscore = jiaodaiscoreItem->getValueAsDouble();
        auto jiaodaiareaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jiaodaiarea$"));
        if (!jiaodaiareaItem) {
            throw std::runtime_error("$variable$jiaodaiarea is not found");
        }
        jiaodaiarea = jiaodaiareaItem->getValueAsDouble();
    }

    inline ScoreConfigSmartCroppingOfBags::ScoreConfigSmartCroppingOfBags(const ScoreConfigSmartCroppingOfBags& obj)
    {
        heiba = obj.heiba;
        heibascore = obj.heibascore;
        heibaarea = obj.heibaarea;
        shudang = obj.shudang;
        shudangscore = obj.shudangscore;
        shudangarea = obj.shudangarea;
        huapo = obj.huapo;
        huaposcore = obj.huaposcore;
        huapoarea = obj.huapoarea;
        jietou = obj.jietou;
        jietouscore = obj.jietouscore;
        jietouarea = obj.jietouarea;
        guasi = obj.guasi;
        guasiscore = obj.guasiscore;
        guasiarea = obj.guasiarea;
        podong = obj.podong;
        podongscore = obj.podongscore;
        podongarea = obj.podongarea;
        zangwu = obj.zangwu;
        zangwuscore = obj.zangwuscore;
        zangwuarea = obj.zangwuarea;
        noshudang = obj.noshudang;
        noshudangscore = obj.noshudangscore;
        noshudangarea = obj.noshudangarea;
        modian = obj.modian;
        modianscore = obj.modianscore;
        modianarea = obj.modianarea;
        loumo = obj.loumo;
        loumoscore = obj.loumoscore;
        loumoarea = obj.loumoarea;
        xishudang = obj.xishudang;
        xishudangscore = obj.xishudangscore;
        xishudangarea = obj.xishudangarea;
        erweima = obj.erweima;
        erweimascore = obj.erweimascore;
        erweimaarea = obj.erweimaarea;
        damodian = obj.damodian;
        damodianscore = obj.damodianscore;
        damodianarea = obj.damodianarea;
        kongdong = obj.kongdong;
        kongdongscore = obj.kongdongscore;
        kongdongarea = obj.kongdongarea;
        sebiao = obj.sebiao;
        sebiaoscore = obj.sebiaoscore;
        sebiaoarea = obj.sebiaoarea;
        yinshuaquexian = obj.yinshuaquexian;
        yinshuaquexianscore = obj.yinshuaquexianscore;
        yinshuaquexianarea = obj.yinshuaquexianarea;
        xiaopodong = obj.xiaopodong;
        xiaopodongscore = obj.xiaopodongscore;
        xiaopodongarea = obj.xiaopodongarea;
        jiaodai = obj.jiaodai;
        jiaodaiscore = obj.jiaodaiscore;
        jiaodaiarea = obj.jiaodaiarea;
    }

    inline ScoreConfigSmartCroppingOfBags& ScoreConfigSmartCroppingOfBags::operator=(const ScoreConfigSmartCroppingOfBags& obj)
    {
        if (this != &obj) {
            heiba = obj.heiba;
            heibascore = obj.heibascore;
            heibaarea = obj.heibaarea;
            shudang = obj.shudang;
            shudangscore = obj.shudangscore;
            shudangarea = obj.shudangarea;
            huapo = obj.huapo;
            huaposcore = obj.huaposcore;
            huapoarea = obj.huapoarea;
            jietou = obj.jietou;
            jietouscore = obj.jietouscore;
            jietouarea = obj.jietouarea;
            guasi = obj.guasi;
            guasiscore = obj.guasiscore;
            guasiarea = obj.guasiarea;
            podong = obj.podong;
            podongscore = obj.podongscore;
            podongarea = obj.podongarea;
            zangwu = obj.zangwu;
            zangwuscore = obj.zangwuscore;
            zangwuarea = obj.zangwuarea;
            noshudang = obj.noshudang;
            noshudangscore = obj.noshudangscore;
            noshudangarea = obj.noshudangarea;
            modian = obj.modian;
            modianscore = obj.modianscore;
            modianarea = obj.modianarea;
            loumo = obj.loumo;
            loumoscore = obj.loumoscore;
            loumoarea = obj.loumoarea;
            xishudang = obj.xishudang;
            xishudangscore = obj.xishudangscore;
            xishudangarea = obj.xishudangarea;
            erweima = obj.erweima;
            erweimascore = obj.erweimascore;
            erweimaarea = obj.erweimaarea;
            damodian = obj.damodian;
            damodianscore = obj.damodianscore;
            damodianarea = obj.damodianarea;
            kongdong = obj.kongdong;
            kongdongscore = obj.kongdongscore;
            kongdongarea = obj.kongdongarea;
            sebiao = obj.sebiao;
            sebiaoscore = obj.sebiaoscore;
            sebiaoarea = obj.sebiaoarea;
            yinshuaquexian = obj.yinshuaquexian;
            yinshuaquexianscore = obj.yinshuaquexianscore;
            yinshuaquexianarea = obj.yinshuaquexianarea;
            xiaopodong = obj.xiaopodong;
            xiaopodongscore = obj.xiaopodongscore;
            xiaopodongarea = obj.xiaopodongarea;
            jiaodai = obj.jiaodai;
            jiaodaiscore = obj.jiaodaiscore;
            jiaodaiarea = obj.jiaodaiarea;
        }
        return *this;
    }

    inline ScoreConfigSmartCroppingOfBags::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ScoreConfigSmartCroppingOfBags$");
        auto heibaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        heibaItem->setName("$variable$heiba$");
        heibaItem->setValueFromBool(heiba);
        assembly.addItem(heibaItem);
        auto heibascoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        heibascoreItem->setName("$variable$heibascore$");
        heibascoreItem->setValueFromDouble(heibascore);
        assembly.addItem(heibascoreItem);
        auto heibaareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        heibaareaItem->setName("$variable$heibaarea$");
        heibaareaItem->setValueFromDouble(heibaarea);
        assembly.addItem(heibaareaItem);
        auto shudangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shudangItem->setName("$variable$shudang$");
        shudangItem->setValueFromBool(shudang);
        assembly.addItem(shudangItem);
        auto shudangscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shudangscoreItem->setName("$variable$shudangscore$");
        shudangscoreItem->setValueFromDouble(shudangscore);
        assembly.addItem(shudangscoreItem);
        auto shudangareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shudangareaItem->setName("$variable$shudangarea$");
        shudangareaItem->setValueFromDouble(shudangarea);
        assembly.addItem(shudangareaItem);
        auto huapoItem = std::make_shared<rw::oso::ObjectStoreItem>();
        huapoItem->setName("$variable$huapo$");
        huapoItem->setValueFromBool(huapo);
        assembly.addItem(huapoItem);
        auto huaposcoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        huaposcoreItem->setName("$variable$huaposcore$");
        huaposcoreItem->setValueFromDouble(huaposcore);
        assembly.addItem(huaposcoreItem);
        auto huapoareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        huapoareaItem->setName("$variable$huapoarea$");
        huapoareaItem->setValueFromDouble(huapoarea);
        assembly.addItem(huapoareaItem);
        auto jietouItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jietouItem->setName("$variable$jietou$");
        jietouItem->setValueFromBool(jietou);
        assembly.addItem(jietouItem);
        auto jietouscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jietouscoreItem->setName("$variable$jietouscore$");
        jietouscoreItem->setValueFromDouble(jietouscore);
        assembly.addItem(jietouscoreItem);
        auto jietouareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jietouareaItem->setName("$variable$jietouarea$");
        jietouareaItem->setValueFromDouble(jietouarea);
        assembly.addItem(jietouareaItem);
        auto guasiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        guasiItem->setName("$variable$guasi$");
        guasiItem->setValueFromBool(guasi);
        assembly.addItem(guasiItem);
        auto guasiscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        guasiscoreItem->setName("$variable$guasiscore$");
        guasiscoreItem->setValueFromDouble(guasiscore);
        assembly.addItem(guasiscoreItem);
        auto guasiareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        guasiareaItem->setName("$variable$guasiarea$");
        guasiareaItem->setValueFromDouble(guasiarea);
        assembly.addItem(guasiareaItem);
        auto podongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        podongItem->setName("$variable$podong$");
        podongItem->setValueFromBool(podong);
        assembly.addItem(podongItem);
        auto podongscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        podongscoreItem->setName("$variable$podongscore$");
        podongscoreItem->setValueFromDouble(podongscore);
        assembly.addItem(podongscoreItem);
        auto podongareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        podongareaItem->setName("$variable$podongarea$");
        podongareaItem->setValueFromDouble(podongarea);
        assembly.addItem(podongareaItem);
        auto zangwuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zangwuItem->setName("$variable$zangwu$");
        zangwuItem->setValueFromBool(zangwu);
        assembly.addItem(zangwuItem);
        auto zangwuscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zangwuscoreItem->setName("$variable$zangwuscore$");
        zangwuscoreItem->setValueFromDouble(zangwuscore);
        assembly.addItem(zangwuscoreItem);
        auto zangwuareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zangwuareaItem->setName("$variable$zangwuarea$");
        zangwuareaItem->setValueFromDouble(zangwuarea);
        assembly.addItem(zangwuareaItem);
        auto noshudangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        noshudangItem->setName("$variable$noshudang$");
        noshudangItem->setValueFromBool(noshudang);
        assembly.addItem(noshudangItem);
        auto noshudangscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        noshudangscoreItem->setName("$variable$noshudangscore$");
        noshudangscoreItem->setValueFromDouble(noshudangscore);
        assembly.addItem(noshudangscoreItem);
        auto noshudangareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        noshudangareaItem->setName("$variable$noshudangarea$");
        noshudangareaItem->setValueFromDouble(noshudangarea);
        assembly.addItem(noshudangareaItem);
        auto modianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        modianItem->setName("$variable$modian$");
        modianItem->setValueFromBool(modian);
        assembly.addItem(modianItem);
        auto modianscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        modianscoreItem->setName("$variable$modianscore$");
        modianscoreItem->setValueFromDouble(modianscore);
        assembly.addItem(modianscoreItem);
        auto modianareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        modianareaItem->setName("$variable$modianarea$");
        modianareaItem->setValueFromDouble(modianarea);
        assembly.addItem(modianareaItem);
        auto loumoItem = std::make_shared<rw::oso::ObjectStoreItem>();
        loumoItem->setName("$variable$loumo$");
        loumoItem->setValueFromBool(loumo);
        assembly.addItem(loumoItem);
        auto loumoscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        loumoscoreItem->setName("$variable$loumoscore$");
        loumoscoreItem->setValueFromDouble(loumoscore);
        assembly.addItem(loumoscoreItem);
        auto loumoareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        loumoareaItem->setName("$variable$loumoarea$");
        loumoareaItem->setValueFromDouble(loumoarea);
        assembly.addItem(loumoareaItem);
        auto xishudangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xishudangItem->setName("$variable$xishudang$");
        xishudangItem->setValueFromBool(xishudang);
        assembly.addItem(xishudangItem);
        auto xishudangscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xishudangscoreItem->setName("$variable$xishudangscore$");
        xishudangscoreItem->setValueFromDouble(xishudangscore);
        assembly.addItem(xishudangscoreItem);
        auto xishudangareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xishudangareaItem->setName("$variable$xishudangarea$");
        xishudangareaItem->setValueFromDouble(xishudangarea);
        assembly.addItem(xishudangareaItem);
        auto erweimaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        erweimaItem->setName("$variable$erweima$");
        erweimaItem->setValueFromBool(erweima);
        assembly.addItem(erweimaItem);
        auto erweimascoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        erweimascoreItem->setName("$variable$erweimascore$");
        erweimascoreItem->setValueFromDouble(erweimascore);
        assembly.addItem(erweimascoreItem);
        auto erweimaareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        erweimaareaItem->setName("$variable$erweimaarea$");
        erweimaareaItem->setValueFromDouble(erweimaarea);
        assembly.addItem(erweimaareaItem);
        auto damodianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        damodianItem->setName("$variable$damodian$");
        damodianItem->setValueFromBool(damodian);
        assembly.addItem(damodianItem);
        auto damodianscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        damodianscoreItem->setName("$variable$damodianscore$");
        damodianscoreItem->setValueFromDouble(damodianscore);
        assembly.addItem(damodianscoreItem);
        auto damodianareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        damodianareaItem->setName("$variable$damodianarea$");
        damodianareaItem->setValueFromDouble(damodianarea);
        assembly.addItem(damodianareaItem);
        auto kongdongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        kongdongItem->setName("$variable$kongdong$");
        kongdongItem->setValueFromBool(kongdong);
        assembly.addItem(kongdongItem);
        auto kongdongscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        kongdongscoreItem->setName("$variable$kongdongscore$");
        kongdongscoreItem->setValueFromDouble(kongdongscore);
        assembly.addItem(kongdongscoreItem);
        auto kongdongareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        kongdongareaItem->setName("$variable$kongdongarea$");
        kongdongareaItem->setValueFromDouble(kongdongarea);
        assembly.addItem(kongdongareaItem);
        auto sebiaoItem = std::make_shared<rw::oso::ObjectStoreItem>();
        sebiaoItem->setName("$variable$sebiao$");
        sebiaoItem->setValueFromBool(sebiao);
        assembly.addItem(sebiaoItem);
        auto sebiaoscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        sebiaoscoreItem->setName("$variable$sebiaoscore$");
        sebiaoscoreItem->setValueFromDouble(sebiaoscore);
        assembly.addItem(sebiaoscoreItem);
        auto sebiaoareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        sebiaoareaItem->setName("$variable$sebiaoarea$");
        sebiaoareaItem->setValueFromDouble(sebiaoarea);
        assembly.addItem(sebiaoareaItem);
        auto yinshuaquexianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yinshuaquexianItem->setName("$variable$yinshuaquexian$");
        yinshuaquexianItem->setValueFromBool(yinshuaquexian);
        assembly.addItem(yinshuaquexianItem);
        auto yinshuaquexianscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yinshuaquexianscoreItem->setName("$variable$yinshuaquexianscore$");
        yinshuaquexianscoreItem->setValueFromDouble(yinshuaquexianscore);
        assembly.addItem(yinshuaquexianscoreItem);
        auto yinshuaquexianareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yinshuaquexianareaItem->setName("$variable$yinshuaquexianarea$");
        yinshuaquexianareaItem->setValueFromDouble(yinshuaquexianarea);
        assembly.addItem(yinshuaquexianareaItem);
        auto xiaopodongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xiaopodongItem->setName("$variable$xiaopodong$");
        xiaopodongItem->setValueFromBool(xiaopodong);
        assembly.addItem(xiaopodongItem);
        auto xiaopodongscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xiaopodongscoreItem->setName("$variable$xiaopodongscore$");
        xiaopodongscoreItem->setValueFromDouble(xiaopodongscore);
        assembly.addItem(xiaopodongscoreItem);
        auto xiaopodongareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xiaopodongareaItem->setName("$variable$xiaopodongarea$");
        xiaopodongareaItem->setValueFromDouble(xiaopodongarea);
        assembly.addItem(xiaopodongareaItem);
        auto jiaodaiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jiaodaiItem->setName("$variable$jiaodai$");
        jiaodaiItem->setValueFromBool(jiaodai);
        assembly.addItem(jiaodaiItem);
        auto jiaodaiscoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jiaodaiscoreItem->setName("$variable$jiaodaiscore$");
        jiaodaiscoreItem->setValueFromDouble(jiaodaiscore);
        assembly.addItem(jiaodaiscoreItem);
        auto jiaodaiareaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jiaodaiareaItem->setName("$variable$jiaodaiarea$");
        jiaodaiareaItem->setValueFromDouble(jiaodaiarea);
        assembly.addItem(jiaodaiareaItem);
        return assembly;
    }

    inline bool ScoreConfigSmartCroppingOfBags::operator==(const ScoreConfigSmartCroppingOfBags& obj) const
    {
        return heiba == obj.heiba && heibascore == obj.heibascore && heibaarea == obj.heibaarea && shudang == obj.shudang && shudangscore == obj.shudangscore && shudangarea == obj.shudangarea && huapo == obj.huapo && huaposcore == obj.huaposcore && huapoarea == obj.huapoarea && jietou == obj.jietou && jietouscore == obj.jietouscore && jietouarea == obj.jietouarea && guasi == obj.guasi && guasiscore == obj.guasiscore && guasiarea == obj.guasiarea && podong == obj.podong && podongscore == obj.podongscore && podongarea == obj.podongarea && zangwu == obj.zangwu && zangwuscore == obj.zangwuscore && zangwuarea == obj.zangwuarea && noshudang == obj.noshudang && noshudangscore == obj.noshudangscore && noshudangarea == obj.noshudangarea && modian == obj.modian && modianscore == obj.modianscore && modianarea == obj.modianarea && loumo == obj.loumo && loumoscore == obj.loumoscore && loumoarea == obj.loumoarea && xishudang == obj.xishudang && xishudangscore == obj.xishudangscore && xishudangarea == obj.xishudangarea && erweima == obj.erweima && erweimascore == obj.erweimascore && erweimaarea == obj.erweimaarea && damodian == obj.damodian && damodianscore == obj.damodianscore && damodianarea == obj.damodianarea && kongdong == obj.kongdong && kongdongscore == obj.kongdongscore && kongdongarea == obj.kongdongarea && sebiao == obj.sebiao && sebiaoscore == obj.sebiaoscore && sebiaoarea == obj.sebiaoarea && yinshuaquexian == obj.yinshuaquexian && yinshuaquexianscore == obj.yinshuaquexianscore && yinshuaquexianarea == obj.yinshuaquexianarea && xiaopodong == obj.xiaopodong && xiaopodongscore == obj.xiaopodongscore && xiaopodongarea == obj.xiaopodongarea && jiaodai == obj.jiaodai && jiaodaiscore == obj.jiaodaiscore && jiaodaiarea == obj.jiaodaiarea;
    }

    inline bool ScoreConfigSmartCroppingOfBags::operator!=(const ScoreConfigSmartCroppingOfBags& obj) const
    {
        return !(*this == obj);
    }

}

