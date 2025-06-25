#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ScoreConfig
    {
    public:
        ScoreConfig() = default;
        ~ScoreConfig() = default;

        ScoreConfig(const rw::oso::ObjectStoreAssembly& assembly);
        ScoreConfig(const ScoreConfig& obj);

        ScoreConfig& operator=(const ScoreConfig& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ScoreConfig& obj) const;
        bool operator!=(const ScoreConfig& obj) const;

    public:
        bool queYa{ false };
        double queYaScore{ 0 };
        double queYaArea{ 0 };
        bool tangShang{ false };
        double tangShangScore{ 0 };
        double tangShangArea{ 0 };
        bool zangWu{ false };
        double zangWuScore{ 0 };
        double zangWuArea{ 0 };
    };

    inline ScoreConfig::ScoreConfig(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ScoreConfig$")
        {
            throw std::runtime_error("Assembly is not $class$ScoreConfig$");
        }
        auto queYaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$queYa$"));
        if (!queYaItem) {
            throw std::runtime_error("$variable$queYa is not found");
        }
        queYa = queYaItem->getValueAsBool();
        auto queYaScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$queYaScore$"));
        if (!queYaScoreItem) {
            throw std::runtime_error("$variable$queYaScore is not found");
        }
        queYaScore = queYaScoreItem->getValueAsDouble();
        auto queYaAreaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$queYaArea$"));
        if (!queYaAreaItem) {
            throw std::runtime_error("$variable$queYaArea is not found");
        }
        queYaArea = queYaAreaItem->getValueAsDouble();
        auto tangShangItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tangShang$"));
        if (!tangShangItem) {
            throw std::runtime_error("$variable$tangShang is not found");
        }
        tangShang = tangShangItem->getValueAsBool();
        auto tangShangScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tangShangScore$"));
        if (!tangShangScoreItem) {
            throw std::runtime_error("$variable$tangShangScore is not found");
        }
        tangShangScore = tangShangScoreItem->getValueAsDouble();
        auto tangShangAreaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$tangShangArea$"));
        if (!tangShangAreaItem) {
            throw std::runtime_error("$variable$tangShangArea is not found");
        }
        tangShangArea = tangShangAreaItem->getValueAsDouble();
        auto zangWuItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zangWu$"));
        if (!zangWuItem) {
            throw std::runtime_error("$variable$zangWu is not found");
        }
        zangWu = zangWuItem->getValueAsBool();
        auto zangWuScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zangWuScore$"));
        if (!zangWuScoreItem) {
            throw std::runtime_error("$variable$zangWuScore is not found");
        }
        zangWuScore = zangWuScoreItem->getValueAsDouble();
        auto zangWuAreaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$zangWuArea$"));
        if (!zangWuAreaItem) {
            throw std::runtime_error("$variable$zangWuArea is not found");
        }
        zangWuArea = zangWuAreaItem->getValueAsDouble();
    }

    inline ScoreConfig::ScoreConfig(const ScoreConfig& obj)
    {
        queYa = obj.queYa;
        queYaScore = obj.queYaScore;
        queYaArea = obj.queYaArea;
        tangShang = obj.tangShang;
        tangShangScore = obj.tangShangScore;
        tangShangArea = obj.tangShangArea;
        zangWu = obj.zangWu;
        zangWuScore = obj.zangWuScore;
        zangWuArea = obj.zangWuArea;
    }

    inline ScoreConfig& ScoreConfig::operator=(const ScoreConfig& obj)
    {
        if (this != &obj) {
            queYa = obj.queYa;
            queYaScore = obj.queYaScore;
            queYaArea = obj.queYaArea;
            tangShang = obj.tangShang;
            tangShangScore = obj.tangShangScore;
            tangShangArea = obj.tangShangArea;
            zangWu = obj.zangWu;
            zangWuScore = obj.zangWuScore;
            zangWuArea = obj.zangWuArea;
        }
        return *this;
    }

    inline ScoreConfig::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ScoreConfig$");
        auto queYaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        queYaItem->setName("$variable$queYa$");
        queYaItem->setValueFromBool(queYa);
        assembly.addItem(queYaItem);
        auto queYaScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        queYaScoreItem->setName("$variable$queYaScore$");
        queYaScoreItem->setValueFromDouble(queYaScore);
        assembly.addItem(queYaScoreItem);
        auto queYaAreaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        queYaAreaItem->setName("$variable$queYaArea$");
        queYaAreaItem->setValueFromDouble(queYaArea);
        assembly.addItem(queYaAreaItem);
        auto tangShangItem = std::make_shared<rw::oso::ObjectStoreItem>();
        tangShangItem->setName("$variable$tangShang$");
        tangShangItem->setValueFromBool(tangShang);
        assembly.addItem(tangShangItem);
        auto tangShangScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        tangShangScoreItem->setName("$variable$tangShangScore$");
        tangShangScoreItem->setValueFromDouble(tangShangScore);
        assembly.addItem(tangShangScoreItem);
        auto tangShangAreaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        tangShangAreaItem->setName("$variable$tangShangArea$");
        tangShangAreaItem->setValueFromDouble(tangShangArea);
        assembly.addItem(tangShangAreaItem);
        auto zangWuItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zangWuItem->setName("$variable$zangWu$");
        zangWuItem->setValueFromBool(zangWu);
        assembly.addItem(zangWuItem);
        auto zangWuScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zangWuScoreItem->setName("$variable$zangWuScore$");
        zangWuScoreItem->setValueFromDouble(zangWuScore);
        assembly.addItem(zangWuScoreItem);
        auto zangWuAreaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        zangWuAreaItem->setName("$variable$zangWuArea$");
        zangWuAreaItem->setValueFromDouble(zangWuArea);
        assembly.addItem(zangWuAreaItem);
        return assembly;
    }

    inline bool ScoreConfig::operator==(const ScoreConfig& obj) const
    {
        return queYa == obj.queYa && queYaScore == obj.queYaScore && queYaArea == obj.queYaArea && tangShang == obj.tangShang && tangShangScore == obj.tangShangScore && tangShangArea == obj.tangShangArea && zangWu == obj.zangWu && zangWuScore == obj.zangWuScore && zangWuArea == obj.zangWuArea;
    }

    inline bool ScoreConfig::operator!=(const ScoreConfig& obj) const
    {
        return !(*this == obj);
    }

}

