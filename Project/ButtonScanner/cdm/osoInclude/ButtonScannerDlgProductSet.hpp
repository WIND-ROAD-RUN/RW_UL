#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class ButtonScannerDlgProductSet
    {
    public:
        ButtonScannerDlgProductSet() = default;
        ~ButtonScannerDlgProductSet() = default;

        ButtonScannerDlgProductSet(const rw::oso::ObjectStoreAssembly& assembly);
        ButtonScannerDlgProductSet(const ButtonScannerDlgProductSet& obj);

        ButtonScannerDlgProductSet& operator=(const ButtonScannerDlgProductSet& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const ButtonScannerDlgProductSet& obj) const;
        bool operator!=(const ButtonScannerDlgProductSet& obj) const;

    public:
        bool outsideDiameterEnable{ false };
        double outsideDiameterValue{ 0 };
        double outsideDiameterDeviation{ 0 };
        double photography{ 0 };
        double blowTime{ 0 };
        bool edgeDamageEnable{ false };
        double edgeDamageSimilarity{ 0 };
        double edgeDamageArea{ 0 };
        bool bengKouEnabel{ false };
        double bengKouScore{ 0 };
        bool shieldingRangeEnable{ false };
        double outerRadius{ 0 };
        double innerRadius{ 0 };
        bool poreEnable{ false };
        double poreEnableScore{ 0 };
        double poreEnableArea{ 0 };
        bool smallPoreEnable{ false };
        double smallPoreEnableScore{ 0 };
        double smallPoreEnableArea{ 0 };
        bool paintEnable{ false };
        double paintEnableScore{ 0 };
        bool holesCountEnable{ false };
        double holesCountValue{ 0 };
        bool brokenEyeEnable{ false };
        double brokenEyeSimilarity{ 0 };
        bool crackEnable{ false };
        double crackSimilarity{ 0 };
        bool apertureEnable{ false };
        double apertureValue{ 0 };
        double apertureSimilarity{ 0 };
        bool holeCenterDistanceEnable{ false };
        double holeCenterDistanceValue{ 0 };
        double holeCenterDistanceSimilarity{ 0 };
        bool specifyColorDifferenceEnable{ false };
        double specifyColorDifferenceR{ 0 };
        double specifyColorDifferenceG{ 0 };
        double specifyColorDifferenceB{ 0 };
        double specifyColorDifferenceDeviation{ 0 };
        bool largeColorDifferenceEnable{ false };
        double largeColorDifferenceDeviation{ 0 };
        bool grindStoneEnable{ false };
        double grindStoneEnableScore{ 0 };
        bool blockEyeEnable{ false };
        double blockEyeEnableScore{ 0 };
        bool materialHeadEnable{ false };
        double materialHeadEnableScore{ 0 };
    };

    inline ButtonScannerDlgProductSet::ButtonScannerDlgProductSet(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$ButtonScannerDlgProductSet$")
        {
            throw std::runtime_error("Assembly is not $class$ButtonScannerDlgProductSet$");
        }
        auto outsideDiameterEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterEnable$"));
        if (!outsideDiameterEnableItem) {
            throw std::runtime_error("$variable$outsideDiameterEnable is not found");
        }
        outsideDiameterEnable = outsideDiameterEnableItem->getValueAsBool();
        auto outsideDiameterValueItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterValue$"));
        if (!outsideDiameterValueItem) {
            throw std::runtime_error("$variable$outsideDiameterValue is not found");
        }
        outsideDiameterValue = outsideDiameterValueItem->getValueAsDouble();
        auto outsideDiameterDeviationItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterDeviation$"));
        if (!outsideDiameterDeviationItem) {
            throw std::runtime_error("$variable$outsideDiameterDeviation is not found");
        }
        outsideDiameterDeviation = outsideDiameterDeviationItem->getValueAsDouble();
        auto photographyItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$photography$"));
        if (!photographyItem) {
            throw std::runtime_error("$variable$photography is not found");
        }
        photography = photographyItem->getValueAsDouble();
        auto blowTimeItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blowTime$"));
        if (!blowTimeItem) {
            throw std::runtime_error("$variable$blowTime is not found");
        }
        blowTime = blowTimeItem->getValueAsDouble();
        auto edgeDamageEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$edgeDamageEnable$"));
        if (!edgeDamageEnableItem) {
            throw std::runtime_error("$variable$edgeDamageEnable is not found");
        }
        edgeDamageEnable = edgeDamageEnableItem->getValueAsBool();
        auto edgeDamageSimilarityItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$edgeDamageSimilarity$"));
        if (!edgeDamageSimilarityItem) {
            throw std::runtime_error("$variable$edgeDamageSimilarity is not found");
        }
        edgeDamageSimilarity = edgeDamageSimilarityItem->getValueAsDouble();
        auto edgeDamageAreaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$edgeDamageArea$"));
        if (!edgeDamageAreaItem) {
            throw std::runtime_error("$variable$edgeDamageArea is not found");
        }
        edgeDamageArea = edgeDamageAreaItem->getValueAsDouble();
        auto bengKouEnabelItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$bengKouEnabel$"));
        if (!bengKouEnabelItem) {
            throw std::runtime_error("$variable$bengKouEnabel is not found");
        }
        bengKouEnabel = bengKouEnabelItem->getValueAsBool();
        auto bengKouScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$bengKouScore$"));
        if (!bengKouScoreItem) {
            throw std::runtime_error("$variable$bengKouScore is not found");
        }
        bengKouScore = bengKouScoreItem->getValueAsDouble();
        auto shieldingRangeEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$shieldingRangeEnable$"));
        if (!shieldingRangeEnableItem) {
            throw std::runtime_error("$variable$shieldingRangeEnable is not found");
        }
        shieldingRangeEnable = shieldingRangeEnableItem->getValueAsBool();
        auto outerRadiusItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outerRadius$"));
        if (!outerRadiusItem) {
            throw std::runtime_error("$variable$outerRadius is not found");
        }
        outerRadius = outerRadiusItem->getValueAsDouble();
        auto innerRadiusItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$innerRadius$"));
        if (!innerRadiusItem) {
            throw std::runtime_error("$variable$innerRadius is not found");
        }
        innerRadius = innerRadiusItem->getValueAsDouble();
        auto poreEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$poreEnable$"));
        if (!poreEnableItem) {
            throw std::runtime_error("$variable$poreEnable is not found");
        }
        poreEnable = poreEnableItem->getValueAsBool();
        auto poreEnableScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$poreEnableScore$"));
        if (!poreEnableScoreItem) {
            throw std::runtime_error("$variable$poreEnableScore is not found");
        }
        poreEnableScore = poreEnableScoreItem->getValueAsDouble();
        auto poreEnableAreaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$poreEnableArea$"));
        if (!poreEnableAreaItem) {
            throw std::runtime_error("$variable$poreEnableArea is not found");
        }
        poreEnableArea = poreEnableAreaItem->getValueAsDouble();
        auto smallPoreEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$smallPoreEnable$"));
        if (!smallPoreEnableItem) {
            throw std::runtime_error("$variable$smallPoreEnable is not found");
        }
        smallPoreEnable = smallPoreEnableItem->getValueAsBool();
        auto smallPoreEnableScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$smallPoreEnableScore$"));
        if (!smallPoreEnableScoreItem) {
            throw std::runtime_error("$variable$smallPoreEnableScore is not found");
        }
        smallPoreEnableScore = smallPoreEnableScoreItem->getValueAsDouble();
        auto smallPoreEnableAreaItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$smallPoreEnableArea$"));
        if (!smallPoreEnableAreaItem) {
            throw std::runtime_error("$variable$smallPoreEnableArea is not found");
        }
        smallPoreEnableArea = smallPoreEnableAreaItem->getValueAsDouble();
        auto paintEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$paintEnable$"));
        if (!paintEnableItem) {
            throw std::runtime_error("$variable$paintEnable is not found");
        }
        paintEnable = paintEnableItem->getValueAsBool();
        auto paintEnableScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$paintEnableScore$"));
        if (!paintEnableScoreItem) {
            throw std::runtime_error("$variable$paintEnableScore is not found");
        }
        paintEnableScore = paintEnableScoreItem->getValueAsDouble();
        auto holesCountEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holesCountEnable$"));
        if (!holesCountEnableItem) {
            throw std::runtime_error("$variable$holesCountEnable is not found");
        }
        holesCountEnable = holesCountEnableItem->getValueAsBool();
        auto holesCountValueItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holesCountValue$"));
        if (!holesCountValueItem) {
            throw std::runtime_error("$variable$holesCountValue is not found");
        }
        holesCountValue = holesCountValueItem->getValueAsDouble();
        auto brokenEyeEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$brokenEyeEnable$"));
        if (!brokenEyeEnableItem) {
            throw std::runtime_error("$variable$brokenEyeEnable is not found");
        }
        brokenEyeEnable = brokenEyeEnableItem->getValueAsBool();
        auto brokenEyeSimilarityItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$brokenEyeSimilarity$"));
        if (!brokenEyeSimilarityItem) {
            throw std::runtime_error("$variable$brokenEyeSimilarity is not found");
        }
        brokenEyeSimilarity = brokenEyeSimilarityItem->getValueAsDouble();
        auto crackEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$crackEnable$"));
        if (!crackEnableItem) {
            throw std::runtime_error("$variable$crackEnable is not found");
        }
        crackEnable = crackEnableItem->getValueAsBool();
        auto crackSimilarityItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$crackSimilarity$"));
        if (!crackSimilarityItem) {
            throw std::runtime_error("$variable$crackSimilarity is not found");
        }
        crackSimilarity = crackSimilarityItem->getValueAsDouble();
        auto apertureEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$apertureEnable$"));
        if (!apertureEnableItem) {
            throw std::runtime_error("$variable$apertureEnable is not found");
        }
        apertureEnable = apertureEnableItem->getValueAsBool();
        auto apertureValueItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$apertureValue$"));
        if (!apertureValueItem) {
            throw std::runtime_error("$variable$apertureValue is not found");
        }
        apertureValue = apertureValueItem->getValueAsDouble();
        auto apertureSimilarityItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$apertureSimilarity$"));
        if (!apertureSimilarityItem) {
            throw std::runtime_error("$variable$apertureSimilarity is not found");
        }
        apertureSimilarity = apertureSimilarityItem->getValueAsDouble();
        auto holeCenterDistanceEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holeCenterDistanceEnable$"));
        if (!holeCenterDistanceEnableItem) {
            throw std::runtime_error("$variable$holeCenterDistanceEnable is not found");
        }
        holeCenterDistanceEnable = holeCenterDistanceEnableItem->getValueAsBool();
        auto holeCenterDistanceValueItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holeCenterDistanceValue$"));
        if (!holeCenterDistanceValueItem) {
            throw std::runtime_error("$variable$holeCenterDistanceValue is not found");
        }
        holeCenterDistanceValue = holeCenterDistanceValueItem->getValueAsDouble();
        auto holeCenterDistanceSimilarityItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$holeCenterDistanceSimilarity$"));
        if (!holeCenterDistanceSimilarityItem) {
            throw std::runtime_error("$variable$holeCenterDistanceSimilarity is not found");
        }
        holeCenterDistanceSimilarity = holeCenterDistanceSimilarityItem->getValueAsDouble();
        auto specifyColorDifferenceEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceEnable$"));
        if (!specifyColorDifferenceEnableItem) {
            throw std::runtime_error("$variable$specifyColorDifferenceEnable is not found");
        }
        specifyColorDifferenceEnable = specifyColorDifferenceEnableItem->getValueAsBool();
        auto specifyColorDifferenceRItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceR$"));
        if (!specifyColorDifferenceRItem) {
            throw std::runtime_error("$variable$specifyColorDifferenceR is not found");
        }
        specifyColorDifferenceR = specifyColorDifferenceRItem->getValueAsDouble();
        auto specifyColorDifferenceGItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceG$"));
        if (!specifyColorDifferenceGItem) {
            throw std::runtime_error("$variable$specifyColorDifferenceG is not found");
        }
        specifyColorDifferenceG = specifyColorDifferenceGItem->getValueAsDouble();
        auto specifyColorDifferenceBItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceB$"));
        if (!specifyColorDifferenceBItem) {
            throw std::runtime_error("$variable$specifyColorDifferenceB is not found");
        }
        specifyColorDifferenceB = specifyColorDifferenceBItem->getValueAsDouble();
        auto specifyColorDifferenceDeviationItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$specifyColorDifferenceDeviation$"));
        if (!specifyColorDifferenceDeviationItem) {
            throw std::runtime_error("$variable$specifyColorDifferenceDeviation is not found");
        }
        specifyColorDifferenceDeviation = specifyColorDifferenceDeviationItem->getValueAsDouble();
        auto largeColorDifferenceEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$largeColorDifferenceEnable$"));
        if (!largeColorDifferenceEnableItem) {
            throw std::runtime_error("$variable$largeColorDifferenceEnable is not found");
        }
        largeColorDifferenceEnable = largeColorDifferenceEnableItem->getValueAsBool();
        auto largeColorDifferenceDeviationItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$largeColorDifferenceDeviation$"));
        if (!largeColorDifferenceDeviationItem) {
            throw std::runtime_error("$variable$largeColorDifferenceDeviation is not found");
        }
        largeColorDifferenceDeviation = largeColorDifferenceDeviationItem->getValueAsDouble();
        auto grindStoneEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$grindStoneEnable$"));
        if (!grindStoneEnableItem) {
            throw std::runtime_error("$variable$grindStoneEnable is not found");
        }
        grindStoneEnable = grindStoneEnableItem->getValueAsBool();
        auto grindStoneEnableScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$grindStoneEnableScore$"));
        if (!grindStoneEnableScoreItem) {
            throw std::runtime_error("$variable$grindStoneEnableScore is not found");
        }
        grindStoneEnableScore = grindStoneEnableScoreItem->getValueAsDouble();
        auto blockEyeEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blockEyeEnable$"));
        if (!blockEyeEnableItem) {
            throw std::runtime_error("$variable$blockEyeEnable is not found");
        }
        blockEyeEnable = blockEyeEnableItem->getValueAsBool();
        auto blockEyeEnableScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$blockEyeEnableScore$"));
        if (!blockEyeEnableScoreItem) {
            throw std::runtime_error("$variable$blockEyeEnableScore is not found");
        }
        blockEyeEnableScore = blockEyeEnableScoreItem->getValueAsDouble();
        auto materialHeadEnableItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$materialHeadEnable$"));
        if (!materialHeadEnableItem) {
            throw std::runtime_error("$variable$materialHeadEnable is not found");
        }
        materialHeadEnable = materialHeadEnableItem->getValueAsBool();
        auto materialHeadEnableScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$materialHeadEnableScore$"));
        if (!materialHeadEnableScoreItem) {
            throw std::runtime_error("$variable$materialHeadEnableScore is not found");
        }
        materialHeadEnableScore = materialHeadEnableScoreItem->getValueAsDouble();
    }

    inline ButtonScannerDlgProductSet::ButtonScannerDlgProductSet(const ButtonScannerDlgProductSet& obj)
    {
        outsideDiameterEnable = obj.outsideDiameterEnable;
        outsideDiameterValue = obj.outsideDiameterValue;
        outsideDiameterDeviation = obj.outsideDiameterDeviation;
        photography = obj.photography;
        blowTime = obj.blowTime;
        edgeDamageEnable = obj.edgeDamageEnable;
        edgeDamageSimilarity = obj.edgeDamageSimilarity;
        edgeDamageArea = obj.edgeDamageArea;
        bengKouEnabel = obj.bengKouEnabel;
        bengKouScore = obj.bengKouScore;
        shieldingRangeEnable = obj.shieldingRangeEnable;
        outerRadius = obj.outerRadius;
        innerRadius = obj.innerRadius;
        poreEnable = obj.poreEnable;
        poreEnableScore = obj.poreEnableScore;
        poreEnableArea = obj.poreEnableArea;
        smallPoreEnable = obj.smallPoreEnable;
        smallPoreEnableScore = obj.smallPoreEnableScore;
        smallPoreEnableArea = obj.smallPoreEnableArea;
        paintEnable = obj.paintEnable;
        paintEnableScore = obj.paintEnableScore;
        holesCountEnable = obj.holesCountEnable;
        holesCountValue = obj.holesCountValue;
        brokenEyeEnable = obj.brokenEyeEnable;
        brokenEyeSimilarity = obj.brokenEyeSimilarity;
        crackEnable = obj.crackEnable;
        crackSimilarity = obj.crackSimilarity;
        apertureEnable = obj.apertureEnable;
        apertureValue = obj.apertureValue;
        apertureSimilarity = obj.apertureSimilarity;
        holeCenterDistanceEnable = obj.holeCenterDistanceEnable;
        holeCenterDistanceValue = obj.holeCenterDistanceValue;
        holeCenterDistanceSimilarity = obj.holeCenterDistanceSimilarity;
        specifyColorDifferenceEnable = obj.specifyColorDifferenceEnable;
        specifyColorDifferenceR = obj.specifyColorDifferenceR;
        specifyColorDifferenceG = obj.specifyColorDifferenceG;
        specifyColorDifferenceB = obj.specifyColorDifferenceB;
        specifyColorDifferenceDeviation = obj.specifyColorDifferenceDeviation;
        largeColorDifferenceEnable = obj.largeColorDifferenceEnable;
        largeColorDifferenceDeviation = obj.largeColorDifferenceDeviation;
        grindStoneEnable = obj.grindStoneEnable;
        grindStoneEnableScore = obj.grindStoneEnableScore;
        blockEyeEnable = obj.blockEyeEnable;
        blockEyeEnableScore = obj.blockEyeEnableScore;
        materialHeadEnable = obj.materialHeadEnable;
        materialHeadEnableScore = obj.materialHeadEnableScore;
    }

    inline ButtonScannerDlgProductSet& ButtonScannerDlgProductSet::operator=(const ButtonScannerDlgProductSet& obj)
    {
        if (this != &obj) {
            outsideDiameterEnable = obj.outsideDiameterEnable;
            outsideDiameterValue = obj.outsideDiameterValue;
            outsideDiameterDeviation = obj.outsideDiameterDeviation;
            photography = obj.photography;
            blowTime = obj.blowTime;
            edgeDamageEnable = obj.edgeDamageEnable;
            edgeDamageSimilarity = obj.edgeDamageSimilarity;
            edgeDamageArea = obj.edgeDamageArea;
            bengKouEnabel = obj.bengKouEnabel;
            bengKouScore = obj.bengKouScore;
            shieldingRangeEnable = obj.shieldingRangeEnable;
            outerRadius = obj.outerRadius;
            innerRadius = obj.innerRadius;
            poreEnable = obj.poreEnable;
            poreEnableScore = obj.poreEnableScore;
            poreEnableArea = obj.poreEnableArea;
            smallPoreEnable = obj.smallPoreEnable;
            smallPoreEnableScore = obj.smallPoreEnableScore;
            smallPoreEnableArea = obj.smallPoreEnableArea;
            paintEnable = obj.paintEnable;
            paintEnableScore = obj.paintEnableScore;
            holesCountEnable = obj.holesCountEnable;
            holesCountValue = obj.holesCountValue;
            brokenEyeEnable = obj.brokenEyeEnable;
            brokenEyeSimilarity = obj.brokenEyeSimilarity;
            crackEnable = obj.crackEnable;
            crackSimilarity = obj.crackSimilarity;
            apertureEnable = obj.apertureEnable;
            apertureValue = obj.apertureValue;
            apertureSimilarity = obj.apertureSimilarity;
            holeCenterDistanceEnable = obj.holeCenterDistanceEnable;
            holeCenterDistanceValue = obj.holeCenterDistanceValue;
            holeCenterDistanceSimilarity = obj.holeCenterDistanceSimilarity;
            specifyColorDifferenceEnable = obj.specifyColorDifferenceEnable;
            specifyColorDifferenceR = obj.specifyColorDifferenceR;
            specifyColorDifferenceG = obj.specifyColorDifferenceG;
            specifyColorDifferenceB = obj.specifyColorDifferenceB;
            specifyColorDifferenceDeviation = obj.specifyColorDifferenceDeviation;
            largeColorDifferenceEnable = obj.largeColorDifferenceEnable;
            largeColorDifferenceDeviation = obj.largeColorDifferenceDeviation;
            grindStoneEnable = obj.grindStoneEnable;
            grindStoneEnableScore = obj.grindStoneEnableScore;
            blockEyeEnable = obj.blockEyeEnable;
            blockEyeEnableScore = obj.blockEyeEnableScore;
            materialHeadEnable = obj.materialHeadEnable;
            materialHeadEnableScore = obj.materialHeadEnableScore;
        }
        return *this;
    }

    inline ButtonScannerDlgProductSet::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$ButtonScannerDlgProductSet$");
        auto outsideDiameterEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        outsideDiameterEnableItem->setName("$variable$outsideDiameterEnable$");
        outsideDiameterEnableItem->setValueFromBool(outsideDiameterEnable);
        assembly.addItem(outsideDiameterEnableItem);
        auto outsideDiameterValueItem = std::make_shared<rw::oso::ObjectStoreItem>();
        outsideDiameterValueItem->setName("$variable$outsideDiameterValue$");
        outsideDiameterValueItem->setValueFromDouble(outsideDiameterValue);
        assembly.addItem(outsideDiameterValueItem);
        auto outsideDiameterDeviationItem = std::make_shared<rw::oso::ObjectStoreItem>();
        outsideDiameterDeviationItem->setName("$variable$outsideDiameterDeviation$");
        outsideDiameterDeviationItem->setValueFromDouble(outsideDiameterDeviation);
        assembly.addItem(outsideDiameterDeviationItem);
        auto photographyItem = std::make_shared<rw::oso::ObjectStoreItem>();
        photographyItem->setName("$variable$photography$");
        photographyItem->setValueFromDouble(photography);
        assembly.addItem(photographyItem);
        auto blowTimeItem = std::make_shared<rw::oso::ObjectStoreItem>();
        blowTimeItem->setName("$variable$blowTime$");
        blowTimeItem->setValueFromDouble(blowTime);
        assembly.addItem(blowTimeItem);
        auto edgeDamageEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        edgeDamageEnableItem->setName("$variable$edgeDamageEnable$");
        edgeDamageEnableItem->setValueFromBool(edgeDamageEnable);
        assembly.addItem(edgeDamageEnableItem);
        auto edgeDamageSimilarityItem = std::make_shared<rw::oso::ObjectStoreItem>();
        edgeDamageSimilarityItem->setName("$variable$edgeDamageSimilarity$");
        edgeDamageSimilarityItem->setValueFromDouble(edgeDamageSimilarity);
        assembly.addItem(edgeDamageSimilarityItem);
        auto edgeDamageAreaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        edgeDamageAreaItem->setName("$variable$edgeDamageArea$");
        edgeDamageAreaItem->setValueFromDouble(edgeDamageArea);
        assembly.addItem(edgeDamageAreaItem);
        auto bengKouEnabelItem = std::make_shared<rw::oso::ObjectStoreItem>();
        bengKouEnabelItem->setName("$variable$bengKouEnabel$");
        bengKouEnabelItem->setValueFromBool(bengKouEnabel);
        assembly.addItem(bengKouEnabelItem);
        auto bengKouScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        bengKouScoreItem->setName("$variable$bengKouScore$");
        bengKouScoreItem->setValueFromDouble(bengKouScore);
        assembly.addItem(bengKouScoreItem);
        auto shieldingRangeEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        shieldingRangeEnableItem->setName("$variable$shieldingRangeEnable$");
        shieldingRangeEnableItem->setValueFromBool(shieldingRangeEnable);
        assembly.addItem(shieldingRangeEnableItem);
        auto outerRadiusItem = std::make_shared<rw::oso::ObjectStoreItem>();
        outerRadiusItem->setName("$variable$outerRadius$");
        outerRadiusItem->setValueFromDouble(outerRadius);
        assembly.addItem(outerRadiusItem);
        auto innerRadiusItem = std::make_shared<rw::oso::ObjectStoreItem>();
        innerRadiusItem->setName("$variable$innerRadius$");
        innerRadiusItem->setValueFromDouble(innerRadius);
        assembly.addItem(innerRadiusItem);
        auto poreEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        poreEnableItem->setName("$variable$poreEnable$");
        poreEnableItem->setValueFromBool(poreEnable);
        assembly.addItem(poreEnableItem);
        auto poreEnableScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        poreEnableScoreItem->setName("$variable$poreEnableScore$");
        poreEnableScoreItem->setValueFromDouble(poreEnableScore);
        assembly.addItem(poreEnableScoreItem);
        auto poreEnableAreaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        poreEnableAreaItem->setName("$variable$poreEnableArea$");
        poreEnableAreaItem->setValueFromDouble(poreEnableArea);
        assembly.addItem(poreEnableAreaItem);
        auto smallPoreEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        smallPoreEnableItem->setName("$variable$smallPoreEnable$");
        smallPoreEnableItem->setValueFromBool(smallPoreEnable);
        assembly.addItem(smallPoreEnableItem);
        auto smallPoreEnableScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        smallPoreEnableScoreItem->setName("$variable$smallPoreEnableScore$");
        smallPoreEnableScoreItem->setValueFromDouble(smallPoreEnableScore);
        assembly.addItem(smallPoreEnableScoreItem);
        auto smallPoreEnableAreaItem = std::make_shared<rw::oso::ObjectStoreItem>();
        smallPoreEnableAreaItem->setName("$variable$smallPoreEnableArea$");
        smallPoreEnableAreaItem->setValueFromDouble(smallPoreEnableArea);
        assembly.addItem(smallPoreEnableAreaItem);
        auto paintEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        paintEnableItem->setName("$variable$paintEnable$");
        paintEnableItem->setValueFromBool(paintEnable);
        assembly.addItem(paintEnableItem);
        auto paintEnableScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        paintEnableScoreItem->setName("$variable$paintEnableScore$");
        paintEnableScoreItem->setValueFromDouble(paintEnableScore);
        assembly.addItem(paintEnableScoreItem);
        auto holesCountEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        holesCountEnableItem->setName("$variable$holesCountEnable$");
        holesCountEnableItem->setValueFromBool(holesCountEnable);
        assembly.addItem(holesCountEnableItem);
        auto holesCountValueItem = std::make_shared<rw::oso::ObjectStoreItem>();
        holesCountValueItem->setName("$variable$holesCountValue$");
        holesCountValueItem->setValueFromDouble(holesCountValue);
        assembly.addItem(holesCountValueItem);
        auto brokenEyeEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        brokenEyeEnableItem->setName("$variable$brokenEyeEnable$");
        brokenEyeEnableItem->setValueFromBool(brokenEyeEnable);
        assembly.addItem(brokenEyeEnableItem);
        auto brokenEyeSimilarityItem = std::make_shared<rw::oso::ObjectStoreItem>();
        brokenEyeSimilarityItem->setName("$variable$brokenEyeSimilarity$");
        brokenEyeSimilarityItem->setValueFromDouble(brokenEyeSimilarity);
        assembly.addItem(brokenEyeSimilarityItem);
        auto crackEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        crackEnableItem->setName("$variable$crackEnable$");
        crackEnableItem->setValueFromBool(crackEnable);
        assembly.addItem(crackEnableItem);
        auto crackSimilarityItem = std::make_shared<rw::oso::ObjectStoreItem>();
        crackSimilarityItem->setName("$variable$crackSimilarity$");
        crackSimilarityItem->setValueFromDouble(crackSimilarity);
        assembly.addItem(crackSimilarityItem);
        auto apertureEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        apertureEnableItem->setName("$variable$apertureEnable$");
        apertureEnableItem->setValueFromBool(apertureEnable);
        assembly.addItem(apertureEnableItem);
        auto apertureValueItem = std::make_shared<rw::oso::ObjectStoreItem>();
        apertureValueItem->setName("$variable$apertureValue$");
        apertureValueItem->setValueFromDouble(apertureValue);
        assembly.addItem(apertureValueItem);
        auto apertureSimilarityItem = std::make_shared<rw::oso::ObjectStoreItem>();
        apertureSimilarityItem->setName("$variable$apertureSimilarity$");
        apertureSimilarityItem->setValueFromDouble(apertureSimilarity);
        assembly.addItem(apertureSimilarityItem);
        auto holeCenterDistanceEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        holeCenterDistanceEnableItem->setName("$variable$holeCenterDistanceEnable$");
        holeCenterDistanceEnableItem->setValueFromBool(holeCenterDistanceEnable);
        assembly.addItem(holeCenterDistanceEnableItem);
        auto holeCenterDistanceValueItem = std::make_shared<rw::oso::ObjectStoreItem>();
        holeCenterDistanceValueItem->setName("$variable$holeCenterDistanceValue$");
        holeCenterDistanceValueItem->setValueFromDouble(holeCenterDistanceValue);
        assembly.addItem(holeCenterDistanceValueItem);
        auto holeCenterDistanceSimilarityItem = std::make_shared<rw::oso::ObjectStoreItem>();
        holeCenterDistanceSimilarityItem->setName("$variable$holeCenterDistanceSimilarity$");
        holeCenterDistanceSimilarityItem->setValueFromDouble(holeCenterDistanceSimilarity);
        assembly.addItem(holeCenterDistanceSimilarityItem);
        auto specifyColorDifferenceEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        specifyColorDifferenceEnableItem->setName("$variable$specifyColorDifferenceEnable$");
        specifyColorDifferenceEnableItem->setValueFromBool(specifyColorDifferenceEnable);
        assembly.addItem(specifyColorDifferenceEnableItem);
        auto specifyColorDifferenceRItem = std::make_shared<rw::oso::ObjectStoreItem>();
        specifyColorDifferenceRItem->setName("$variable$specifyColorDifferenceR$");
        specifyColorDifferenceRItem->setValueFromDouble(specifyColorDifferenceR);
        assembly.addItem(specifyColorDifferenceRItem);
        auto specifyColorDifferenceGItem = std::make_shared<rw::oso::ObjectStoreItem>();
        specifyColorDifferenceGItem->setName("$variable$specifyColorDifferenceG$");
        specifyColorDifferenceGItem->setValueFromDouble(specifyColorDifferenceG);
        assembly.addItem(specifyColorDifferenceGItem);
        auto specifyColorDifferenceBItem = std::make_shared<rw::oso::ObjectStoreItem>();
        specifyColorDifferenceBItem->setName("$variable$specifyColorDifferenceB$");
        specifyColorDifferenceBItem->setValueFromDouble(specifyColorDifferenceB);
        assembly.addItem(specifyColorDifferenceBItem);
        auto specifyColorDifferenceDeviationItem = std::make_shared<rw::oso::ObjectStoreItem>();
        specifyColorDifferenceDeviationItem->setName("$variable$specifyColorDifferenceDeviation$");
        specifyColorDifferenceDeviationItem->setValueFromDouble(specifyColorDifferenceDeviation);
        assembly.addItem(specifyColorDifferenceDeviationItem);
        auto largeColorDifferenceEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        largeColorDifferenceEnableItem->setName("$variable$largeColorDifferenceEnable$");
        largeColorDifferenceEnableItem->setValueFromBool(largeColorDifferenceEnable);
        assembly.addItem(largeColorDifferenceEnableItem);
        auto largeColorDifferenceDeviationItem = std::make_shared<rw::oso::ObjectStoreItem>();
        largeColorDifferenceDeviationItem->setName("$variable$largeColorDifferenceDeviation$");
        largeColorDifferenceDeviationItem->setValueFromDouble(largeColorDifferenceDeviation);
        assembly.addItem(largeColorDifferenceDeviationItem);
        auto grindStoneEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        grindStoneEnableItem->setName("$variable$grindStoneEnable$");
        grindStoneEnableItem->setValueFromBool(grindStoneEnable);
        assembly.addItem(grindStoneEnableItem);
        auto grindStoneEnableScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        grindStoneEnableScoreItem->setName("$variable$grindStoneEnableScore$");
        grindStoneEnableScoreItem->setValueFromDouble(grindStoneEnableScore);
        assembly.addItem(grindStoneEnableScoreItem);
        auto blockEyeEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        blockEyeEnableItem->setName("$variable$blockEyeEnable$");
        blockEyeEnableItem->setValueFromBool(blockEyeEnable);
        assembly.addItem(blockEyeEnableItem);
        auto blockEyeEnableScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        blockEyeEnableScoreItem->setName("$variable$blockEyeEnableScore$");
        blockEyeEnableScoreItem->setValueFromDouble(blockEyeEnableScore);
        assembly.addItem(blockEyeEnableScoreItem);
        auto materialHeadEnableItem = std::make_shared<rw::oso::ObjectStoreItem>();
        materialHeadEnableItem->setName("$variable$materialHeadEnable$");
        materialHeadEnableItem->setValueFromBool(materialHeadEnable);
        assembly.addItem(materialHeadEnableItem);
        auto materialHeadEnableScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        materialHeadEnableScoreItem->setName("$variable$materialHeadEnableScore$");
        materialHeadEnableScoreItem->setValueFromDouble(materialHeadEnableScore);
        assembly.addItem(materialHeadEnableScoreItem);
        return assembly;
    }

    inline bool ButtonScannerDlgProductSet::operator==(const ButtonScannerDlgProductSet& obj) const
    {
        return outsideDiameterEnable == obj.outsideDiameterEnable && outsideDiameterValue == obj.outsideDiameterValue && outsideDiameterDeviation == obj.outsideDiameterDeviation && photography == obj.photography && blowTime == obj.blowTime && edgeDamageEnable == obj.edgeDamageEnable && edgeDamageSimilarity == obj.edgeDamageSimilarity && edgeDamageArea == obj.edgeDamageArea && bengKouEnabel == obj.bengKouEnabel && bengKouScore == obj.bengKouScore && shieldingRangeEnable == obj.shieldingRangeEnable && outerRadius == obj.outerRadius && innerRadius == obj.innerRadius && poreEnable == obj.poreEnable && poreEnableScore == obj.poreEnableScore && poreEnableArea == obj.poreEnableArea && smallPoreEnable == obj.smallPoreEnable && smallPoreEnableScore == obj.smallPoreEnableScore && smallPoreEnableArea == obj.smallPoreEnableArea && paintEnable == obj.paintEnable && paintEnableScore == obj.paintEnableScore && holesCountEnable == obj.holesCountEnable && holesCountValue == obj.holesCountValue && brokenEyeEnable == obj.brokenEyeEnable && brokenEyeSimilarity == obj.brokenEyeSimilarity && crackEnable == obj.crackEnable && crackSimilarity == obj.crackSimilarity && apertureEnable == obj.apertureEnable && apertureValue == obj.apertureValue && apertureSimilarity == obj.apertureSimilarity && holeCenterDistanceEnable == obj.holeCenterDistanceEnable && holeCenterDistanceValue == obj.holeCenterDistanceValue && holeCenterDistanceSimilarity == obj.holeCenterDistanceSimilarity && specifyColorDifferenceEnable == obj.specifyColorDifferenceEnable && specifyColorDifferenceR == obj.specifyColorDifferenceR && specifyColorDifferenceG == obj.specifyColorDifferenceG && specifyColorDifferenceB == obj.specifyColorDifferenceB && specifyColorDifferenceDeviation == obj.specifyColorDifferenceDeviation && largeColorDifferenceEnable == obj.largeColorDifferenceEnable && largeColorDifferenceDeviation == obj.largeColorDifferenceDeviation && grindStoneEnable == obj.grindStoneEnable && grindStoneEnableScore == obj.grindStoneEnableScore && blockEyeEnable == obj.blockEyeEnable && blockEyeEnableScore == obj.blockEyeEnableScore && materialHeadEnable == obj.materialHeadEnable && materialHeadEnableScore == obj.materialHeadEnableScore;
    }

    inline bool ButtonScannerDlgProductSet::operator!=(const ButtonScannerDlgProductSet& obj) const
    {
        return !(*this == obj);
    }

}

