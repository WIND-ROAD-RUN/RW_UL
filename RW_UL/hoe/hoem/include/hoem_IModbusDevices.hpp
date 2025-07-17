#pragma once

#include"hoem_utilty.hpp"


namespace rw {
	namespace hoem {
        class IModbusDevice {
        public:
            virtual ~IModbusDevice() = default;

            // 初始化设备连接
            virtual bool connect() = 0;

            // 断开设备连接
            virtual bool disconnect() = 0;

			virtual bool reconnect() = 0;

            virtual bool isConnected() const = 0;

			virtual bool getIState(ModbusI locate) const = 0;

            virtual bool setOState(ModbusO locate, bool state)=0;

            virtual bool getOState(ModbusO locate) const = 0;
        };
	
	}
}