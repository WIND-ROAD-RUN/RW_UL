#pragma once

#include"oso_core.h"
#include <string>

namespace cdm {
    class DlgHideScoreSet
    {
    public:
        DlgHideScoreSet() = default;
        ~DlgHideScoreSet() = default;

        DlgHideScoreSet(const rw::oso::ObjectStoreAssembly& assembly);
        DlgHideScoreSet(const DlgHideScoreSet& obj);

        DlgHideScoreSet& operator=(const DlgHideScoreSet& obj);
        operator rw::oso::ObjectStoreAssembly() const;
        bool operator==(const DlgHideScoreSet& obj) const;
        bool operator!=(const DlgHideScoreSet& obj) const;

    public:
        double outsideDiameterScore{ 0 };
        double forAndAgainstScore{ 0 };
    };

    inline DlgHideScoreSet::DlgHideScoreSet(const rw::oso::ObjectStoreAssembly& assembly)
    {
        auto isAccountAssembly = assembly.getName();
        if (isAccountAssembly != "$class$DlgHideScoreSet$")
        {
            throw std::runtime_error("Assembly is not $class$DlgHideScoreSet$");
        }
        auto outsideDiameterScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$outsideDiameterScore$"));
        if (!outsideDiameterScoreItem) {
            throw std::runtime_error("$variable$outsideDiameterScore is not found");
        }
        outsideDiameterScore = outsideDiameterScoreItem->getValueAsDouble();
        auto forAndAgainstScoreItem = rw::oso::ObjectStoreCoreToItem(assembly.getItem("$variable$forAndAgainstScore$"));
        if (!forAndAgainstScoreItem) {
            throw std::runtime_error("$variable$forAndAgainstScore is not found");
        }
        forAndAgainstScore = forAndAgainstScoreItem->getValueAsDouble();
    }

    inline DlgHideScoreSet::DlgHideScoreSet(const DlgHideScoreSet& obj)
    {
        outsideDiameterScore = obj.outsideDiameterScore;
        forAndAgainstScore = obj.forAndAgainstScore;
    }

    inline DlgHideScoreSet& DlgHideScoreSet::operator=(const DlgHideScoreSet& obj)
    {
        if (this != &obj) {
            outsideDiameterScore = obj.outsideDiameterScore;
            forAndAgainstScore = obj.forAndAgainstScore;
        }
        return *this;
    }

    inline DlgHideScoreSet::operator rw::oso::ObjectStoreAssembly() const
    {
        rw::oso::ObjectStoreAssembly assembly;
        assembly.setName("$class$DlgHideScoreSet$");
        auto outsideDiameterScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        outsideDiameterScoreItem->setName("$variable$outsideDiameterScore$");
        outsideDiameterScoreItem->setValueFromDouble(outsideDiameterScore);
        assembly.addItem(outsideDiameterScoreItem);
        auto forAndAgainstScoreItem = std::make_shared<rw::oso::ObjectStoreItem>();
        forAndAgainstScoreItem->setName("$variable$forAndAgainstScore$");
        forAndAgainstScoreItem->setValueFromDouble(forAndAgainstScore);
        assembly.addItem(forAndAgainstScoreItem);
        return assembly;
    }

    inline bool DlgHideScoreSet::operator==(const DlgHideScoreSet& obj) const
    {
        return outsideDiameterScore == obj.outsideDiameterScore && forAndAgainstScore == obj.forAndAgainstScore;
    }

    inline bool DlgHideScoreSet::operator!=(const DlgHideScoreSet& obj) const
    {
        return !(*this == obj);
    }

}

