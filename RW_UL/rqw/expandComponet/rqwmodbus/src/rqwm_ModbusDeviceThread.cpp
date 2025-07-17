#include "rqwm_ModbusDeviceThread.hpp"

namespace rw::rqwm {

    ModbusDeviceThreadSafe::ModbusDeviceThreadSafe(const ModbusType& type, const ModbusConfig& config, QObject* parent)
        : QObject(parent), _device(std::make_unique<ModbusDevice>(type, config)) {
    }

    ModbusDeviceThreadSafe::~ModbusDeviceThreadSafe() {
        _device.reset();
    }

    bool ModbusDeviceThreadSafe::connect() {
        std::unique_lock lock(_mutex);
        return _device->connect();
    }

    bool ModbusDeviceThreadSafe::disconnect() {
        std::unique_lock lock(_mutex);
        return _device->disconnect();
    }

    bool ModbusDeviceThreadSafe::reconnect() {
        std::unique_lock lock(_mutex); 
        return _device->reconnect();
    }

    bool ModbusDeviceThreadSafe::isConnected() const {
        std::shared_lock lock(_mutex); 
        return _device->isConnected();
    }

    bool ModbusDeviceThreadSafe::getIState(ModbusI locate) const {
        std::shared_lock lock(_mutex); 
        return _device->getIState(locate);
    }

    bool ModbusDeviceThreadSafe::setOState(ModbusO locate, bool state) {
        std::unique_lock lock(_mutex);
        return _device->setOState(locate, state);
    }

    bool ModbusDeviceThreadSafe::getOState(ModbusO locate) const {
        std::shared_lock lock(_mutex); 
        return _device->getOState(locate);
    }

} // namespace rw::rqwm