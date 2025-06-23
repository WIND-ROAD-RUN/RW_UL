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
        bool qiyongerxiangji{ false };
        bool qiyongyundongkongzhiqi{ false };
        bool yundongkongzhiqichonglian{ false };
        double shedingladaichangdu{ 0 };
        double chongkongjishu{ 0 };
        double dangqianchangdu{ 0 };
        double xiangjichufachangdu{ 0 };
        double shoudongsudu{ 0 };
        double meizhuanmaichongshu{ 0 };
        double zidongladaisudu{ 0 };
        double shedingzhouchang{ 0 };
        double yanshichongkong{ 0 };
        double chongkongshijian{ 0 };
        double yanshiziqi{ 0 };
        double jiajiansushijian{ 0 };
        int setqidonganniu{ 0 };
        int setlalianlawan{ 0 };
        int setjiting{ 0 };
        int setbujindianjimaichong{ 0 };
        int setchongkong{ 0 };
        int settuoji{ 0 };
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
        auto takeWork1PicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeWork1Pictures$"));
        if (!takeWork1PicturesItem) {
            throw std::runtime_error("$variable$takeWork1Pictures is not found");
        }
        takeWork1Pictures = takeWork1PicturesItem->getValueAsBool();
        auto takeWork2PicturesItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$takeWork2Pictures$"));
        if (!takeWork2PicturesItem) {
            throw std::runtime_error("$variable$takeWork2Pictures is not found");
        }
        takeWork2Pictures = takeWork2PicturesItem->getValueAsBool();
        auto qiyongerxiangjiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiyongerxiangji$"));
        if (!qiyongerxiangjiItem) {
            throw std::runtime_error("$variable$qiyongerxiangji is not found");
        }
        qiyongerxiangji = qiyongerxiangjiItem->getValueAsBool();
        auto qiyongyundongkongzhiqiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$qiyongyundongkongzhiqi$"));
        if (!qiyongyundongkongzhiqiItem) {
            throw std::runtime_error("$variable$qiyongyundongkongzhiqi is not found");
        }
        qiyongyundongkongzhiqi = qiyongyundongkongzhiqiItem->getValueAsBool();
        auto yundongkongzhiqichonglianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yundongkongzhiqichonglian$"));
        if (!yundongkongzhiqichonglianItem) {
            throw std::runtime_error("$variable$yundongkongzhiqichonglian is not found");
        }
        yundongkongzhiqichonglian = yundongkongzhiqichonglianItem->getValueAsBool();
        auto shedingladaichangduItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shedingladaichangdu$"));
        if (!shedingladaichangduItem) {
            throw std::runtime_error("$variable$shedingladaichangdu is not found");
        }
        shedingladaichangdu = shedingladaichangduItem->getValueAsDouble();
        auto chongkongjishuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chongkongjishu$"));
        if (!chongkongjishuItem) {
            throw std::runtime_error("$variable$chongkongjishu is not found");
        }
        chongkongjishu = chongkongjishuItem->getValueAsDouble();
        auto dangqianchangduItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$dangqianchangdu$"));
        if (!dangqianchangduItem) {
            throw std::runtime_error("$variable$dangqianchangdu is not found");
        }
        dangqianchangdu = dangqianchangduItem->getValueAsDouble();
        auto xiangjichufachangduItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$xiangjichufachangdu$"));
        if (!xiangjichufachangduItem) {
            throw std::runtime_error("$variable$xiangjichufachangdu is not found");
        }
        xiangjichufachangdu = xiangjichufachangduItem->getValueAsDouble();
        auto shoudongsuduItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shoudongsudu$"));
        if (!shoudongsuduItem) {
            throw std::runtime_error("$variable$shoudongsudu is not found");
        }
        shoudongsudu = shoudongsuduItem->getValueAsDouble();
        auto meizhuanmaichongshuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$meizhuanmaichongshu$"));
        if (!meizhuanmaichongshuItem) {
            throw std::runtime_error("$variable$meizhuanmaichongshu is not found");
        }
        meizhuanmaichongshu = meizhuanmaichongshuItem->getValueAsDouble();
        auto zidongladaisuduItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zidongladaisudu$"));
        if (!zidongladaisuduItem) {
            throw std::runtime_error("$variable$zidongladaisudu is not found");
        }
        zidongladaisudu = zidongladaisuduItem->getValueAsDouble();
        auto shedingzhouchangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shedingzhouchang$"));
        if (!shedingzhouchangItem) {
            throw std::runtime_error("$variable$shedingzhouchang is not found");
        }
        shedingzhouchang = shedingzhouchangItem->getValueAsDouble();
        auto yanshichongkongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yanshichongkong$"));
        if (!yanshichongkongItem) {
            throw std::runtime_error("$variable$yanshichongkong is not found");
        }
        yanshichongkong = yanshichongkongItem->getValueAsDouble();
        auto chongkongshijianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$chongkongshijian$"));
        if (!chongkongshijianItem) {
            throw std::runtime_error("$variable$chongkongshijian is not found");
        }
        chongkongshijian = chongkongshijianItem->getValueAsDouble();
        auto yanshiziqiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$yanshiziqi$"));
        if (!yanshiziqiItem) {
            throw std::runtime_error("$variable$yanshiziqi is not found");
        }
        yanshiziqi = yanshiziqiItem->getValueAsDouble();
        auto jiajiansushijianItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$jiajiansushijian$"));
        if (!jiajiansushijianItem) {
            throw std::runtime_error("$variable$jiajiansushijian is not found");
        }
        jiajiansushijian = jiajiansushijianItem->getValueAsDouble();
        auto setqidonganniuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$setqidonganniu$"));
        if (!setqidonganniuItem) {
            throw std::runtime_error("$variable$setqidonganniu is not found");
        }
        setqidonganniu = setqidonganniuItem->getValueAsInt();
        auto setlalianlawanItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$setlalianlawan$"));
        if (!setlalianlawanItem) {
            throw std::runtime_error("$variable$setlalianlawan is not found");
        }
        setlalianlawan = setlalianlawanItem->getValueAsInt();
        auto setjitingItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$setjiting$"));
        if (!setjitingItem) {
            throw std::runtime_error("$variable$setjiting is not found");
        }
        setjiting = setjitingItem->getValueAsInt();
        auto setbujindianjimaichongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$setbujindianjimaichong$"));
        if (!setbujindianjimaichongItem) {
            throw std::runtime_error("$variable$setbujindianjimaichong is not found");
        }
        setbujindianjimaichong = setbujindianjimaichongItem->getValueAsInt();
        auto setchongkongItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$setchongkong$"));
        if (!setchongkongItem) {
            throw std::runtime_error("$variable$setchongkong is not found");
        }
        setchongkong = setchongkongItem->getValueAsInt();
        auto settuojiItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$settuoji$"));
        if (!settuojiItem) {
            throw std::runtime_error("$variable$settuoji is not found");
        }
        settuoji = settuojiItem->getValueAsInt();
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
        takeWork1Pictures = obj.takeWork1Pictures;
        takeWork2Pictures = obj.takeWork2Pictures;
        qiyongerxiangji = obj.qiyongerxiangji;
        qiyongyundongkongzhiqi = obj.qiyongyundongkongzhiqi;
        yundongkongzhiqichonglian = obj.yundongkongzhiqichonglian;
        shedingladaichangdu = obj.shedingladaichangdu;
        chongkongjishu = obj.chongkongjishu;
        dangqianchangdu = obj.dangqianchangdu;
        xiangjichufachangdu = obj.xiangjichufachangdu;
        shoudongsudu = obj.shoudongsudu;
        meizhuanmaichongshu = obj.meizhuanmaichongshu;
        zidongladaisudu = obj.zidongladaisudu;
        shedingzhouchang = obj.shedingzhouchang;
        yanshichongkong = obj.yanshichongkong;
        chongkongshijian = obj.chongkongshijian;
        yanshiziqi = obj.yanshiziqi;
        jiajiansushijian = obj.jiajiansushijian;
        setqidonganniu = obj.setqidonganniu;
        setlalianlawan = obj.setlalianlawan;
        setjiting = obj.setjiting;
        setbujindianjimaichong = obj.setbujindianjimaichong;
        setchongkong = obj.setchongkong;
        settuoji = obj.settuoji;
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
            takeWork1Pictures = obj.takeWork1Pictures;
            takeWork2Pictures = obj.takeWork2Pictures;
            qiyongerxiangji = obj.qiyongerxiangji;
            qiyongyundongkongzhiqi = obj.qiyongyundongkongzhiqi;
            yundongkongzhiqichonglian = obj.yundongkongzhiqichonglian;
            shedingladaichangdu = obj.shedingladaichangdu;
            chongkongjishu = obj.chongkongjishu;
            dangqianchangdu = obj.dangqianchangdu;
            xiangjichufachangdu = obj.xiangjichufachangdu;
            shoudongsudu = obj.shoudongsudu;
            meizhuanmaichongshu = obj.meizhuanmaichongshu;
            zidongladaisudu = obj.zidongladaisudu;
            shedingzhouchang = obj.shedingzhouchang;
            yanshichongkong = obj.yanshichongkong;
            chongkongshijian = obj.chongkongshijian;
            yanshiziqi = obj.yanshiziqi;
            jiajiansushijian = obj.jiajiansushijian;
            setqidonganniu = obj.setqidonganniu;
            setlalianlawan = obj.setlalianlawan;
            setjiting = obj.setjiting;
            setbujindianjimaichong = obj.setbujindianjimaichong;
            setchongkong = obj.setchongkong;
            settuoji = obj.settuoji;
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
        auto takeWork1PicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeWork1PicturesItem->setName("$variable$takeWork1Pictures$");
        takeWork1PicturesItem->setValueFromBool(takeWork1Pictures);
        assembly.addItem(takeWork1PicturesItem);
        auto takeWork2PicturesItem = std::make_shared<rw::oso::ObjectStoreItem>();
        takeWork2PicturesItem->setName("$variable$takeWork2Pictures$");
        takeWork2PicturesItem->setValueFromBool(takeWork2Pictures);
        assembly.addItem(takeWork2PicturesItem);
        auto qiyongerxiangjiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiyongerxiangjiItem->setName("$variable$qiyongerxiangji$");
        qiyongerxiangjiItem->setValueFromBool(qiyongerxiangji);
        assembly.addItem(qiyongerxiangjiItem);
        auto qiyongyundongkongzhiqiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        qiyongyundongkongzhiqiItem->setName("$variable$qiyongyundongkongzhiqi$");
        qiyongyundongkongzhiqiItem->setValueFromBool(qiyongyundongkongzhiqi);
        assembly.addItem(qiyongyundongkongzhiqiItem);
        auto yundongkongzhiqichonglianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yundongkongzhiqichonglianItem->setName("$variable$yundongkongzhiqichonglian$");
        yundongkongzhiqichonglianItem->setValueFromBool(yundongkongzhiqichonglian);
        assembly.addItem(yundongkongzhiqichonglianItem);
        auto shedingladaichangduItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shedingladaichangduItem->setName("$variable$shedingladaichangdu$");
        shedingladaichangduItem->setValueFromDouble(shedingladaichangdu);
        assembly.addItem(shedingladaichangduItem);
        auto chongkongjishuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        chongkongjishuItem->setName("$variable$chongkongjishu$");
        chongkongjishuItem->setValueFromDouble(chongkongjishu);
        assembly.addItem(chongkongjishuItem);
        auto dangqianchangduItem = std::make_shared<rw::oso::ObjectStoreItem>();
        dangqianchangduItem->setName("$variable$dangqianchangdu$");
        dangqianchangduItem->setValueFromDouble(dangqianchangdu);
        assembly.addItem(dangqianchangduItem);
        auto xiangjichufachangduItem = std::make_shared<rw::oso::ObjectStoreItem>();
        xiangjichufachangduItem->setName("$variable$xiangjichufachangdu$");
        xiangjichufachangduItem->setValueFromDouble(xiangjichufachangdu);
        assembly.addItem(xiangjichufachangduItem);
        auto shoudongsuduItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shoudongsuduItem->setName("$variable$shoudongsudu$");
        shoudongsuduItem->setValueFromDouble(shoudongsudu);
        assembly.addItem(shoudongsuduItem);
        auto meizhuanmaichongshuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        meizhuanmaichongshuItem->setName("$variable$meizhuanmaichongshu$");
        meizhuanmaichongshuItem->setValueFromDouble(meizhuanmaichongshu);
        assembly.addItem(meizhuanmaichongshuItem);
        auto zidongladaisuduItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zidongladaisuduItem->setName("$variable$zidongladaisudu$");
        zidongladaisuduItem->setValueFromDouble(zidongladaisudu);
        assembly.addItem(zidongladaisuduItem);
        auto shedingzhouchangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shedingzhouchangItem->setName("$variable$shedingzhouchang$");
        shedingzhouchangItem->setValueFromDouble(shedingzhouchang);
        assembly.addItem(shedingzhouchangItem);
        auto yanshichongkongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yanshichongkongItem->setName("$variable$yanshichongkong$");
        yanshichongkongItem->setValueFromDouble(yanshichongkong);
        assembly.addItem(yanshichongkongItem);
        auto chongkongshijianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        chongkongshijianItem->setName("$variable$chongkongshijian$");
        chongkongshijianItem->setValueFromDouble(chongkongshijian);
        assembly.addItem(chongkongshijianItem);
        auto yanshiziqiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        yanshiziqiItem->setName("$variable$yanshiziqi$");
        yanshiziqiItem->setValueFromDouble(yanshiziqi);
        assembly.addItem(yanshiziqiItem);
        auto jiajiansushijianItem = std::make_shared<rw::oso::ObjectStoreItem>();
        jiajiansushijianItem->setName("$variable$jiajiansushijian$");
        jiajiansushijianItem->setValueFromDouble(jiajiansushijian);
        assembly.addItem(jiajiansushijianItem);
        auto setqidonganniuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        setqidonganniuItem->setName("$variable$setqidonganniu$");
        setqidonganniuItem->setValueFromInt(setqidonganniu);
        assembly.addItem(setqidonganniuItem);
        auto setlalianlawanItem = std::make_shared<rw::oso::ObjectStoreItem>();
        setlalianlawanItem->setName("$variable$setlalianlawan$");
        setlalianlawanItem->setValueFromInt(setlalianlawan);
        assembly.addItem(setlalianlawanItem);
        auto setjitingItem = std::make_shared<rw::oso::ObjectStoreItem>();
        setjitingItem->setName("$variable$setjiting$");
        setjitingItem->setValueFromInt(setjiting);
        assembly.addItem(setjitingItem);
        auto setbujindianjimaichongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        setbujindianjimaichongItem->setName("$variable$setbujindianjimaichong$");
        setbujindianjimaichongItem->setValueFromInt(setbujindianjimaichong);
        assembly.addItem(setbujindianjimaichongItem);
        auto setchongkongItem = std::make_shared<rw::oso::ObjectStoreItem>();
        setchongkongItem->setName("$variable$setchongkong$");
        setchongkongItem->setValueFromInt(setchongkong);
        assembly.addItem(setchongkongItem);
        auto settuojiItem = std::make_shared<rw::oso::ObjectStoreItem>();
        settuojiItem->setName("$variable$settuoji$");
        settuojiItem->setValueFromInt(settuoji);
        assembly.addItem(settuojiItem);
        return assembly;
    }

    inline bool SetConfig::operator==(const SetConfig& obj) const
    {
        return tiFeiChiXuShiJian1 == obj.tiFeiChiXuShiJian1 && yanChiTiFeiShiJian1 == obj.yanChiTiFeiShiJian1 && tiFeiChiXuShiJian2 == obj.tiFeiChiXuShiJian2 && yanChiTiFeiShiJian2 == obj.yanChiTiFeiShiJian2 && shangXianWei1 == obj.shangXianWei1 && xiaXianWei1 == obj.xiaXianWei1 && zuoXianWei1 == obj.zuoXianWei1 && youXianWei1 == obj.youXianWei1 && xiangSuDangLiang1 == obj.xiangSuDangLiang1 && shangXianWei2 == obj.shangXianWei2 && xiaXianWei2 == obj.xiaXianWei2 && zuoXianWei2 == obj.zuoXianWei2 && youXianWei2 == obj.youXianWei2 && xiangSuDangLiang2 == obj.xiangSuDangLiang2 && qiangBaoGuang == obj.qiangBaoGuang && qiangZengYi == obj.qiangZengYi && zhongBaoGuang == obj.zhongBaoGuang && zhongZengYi == obj.zhongZengYi && ruoBaoGuang == obj.ruoBaoGuang && ruoZengYi == obj.ruoZengYi && saveNGImg == obj.saveNGImg && saveMaskImg == obj.saveMaskImg && saveOKImg == obj.saveOKImg && debugMode == obj.debugMode && takeWork1Pictures == obj.takeWork1Pictures && takeWork2Pictures == obj.takeWork2Pictures && qiyongerxiangji == obj.qiyongerxiangji && qiyongyundongkongzhiqi == obj.qiyongyundongkongzhiqi && yundongkongzhiqichonglian == obj.yundongkongzhiqichonglian && shedingladaichangdu == obj.shedingladaichangdu && chongkongjishu == obj.chongkongjishu && dangqianchangdu == obj.dangqianchangdu && xiangjichufachangdu == obj.xiangjichufachangdu && shoudongsudu == obj.shoudongsudu && meizhuanmaichongshu == obj.meizhuanmaichongshu && zidongladaisudu == obj.zidongladaisudu && shedingzhouchang == obj.shedingzhouchang && yanshichongkong == obj.yanshichongkong && chongkongshijian == obj.chongkongshijian && yanshiziqi == obj.yanshiziqi && jiajiansushijian == obj.jiajiansushijian && setqidonganniu == obj.setqidonganniu && setlalianlawan == obj.setlalianlawan && setjiting == obj.setjiting && setbujindianjimaichong == obj.setbujindianjimaichong && setchongkong == obj.setchongkong && settuoji == obj.settuoji;
    }

    inline bool SetConfig::operator!=(const SetConfig& obj) const
    {
        return !(*this == obj);
    }

}

