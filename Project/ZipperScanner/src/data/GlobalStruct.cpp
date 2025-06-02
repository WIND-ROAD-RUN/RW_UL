#include"GlobalStruct.hpp"

GlobalStructDataZipper::GlobalStructDataZipper()
{
}
void GlobalStructDataZipper::buildConfigManager(rw::oso::StorageType type)
{
	storeContext = std::make_unique<rw::oso::StorageContext>(type);
}

