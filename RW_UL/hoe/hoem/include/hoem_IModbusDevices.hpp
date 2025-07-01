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
            virtual void disconnect() = 0;

            virtual bool isConnected() const = 0;

            virtual bool setBasedAddress(Address basedAddress)=0;

			virtual Address getBasedAddress() const = 0;

            virtual bool readRegistersAbsolute(Address address, Quantity quantity, std::vector<RegisterValue>& data) = 0;

            virtual bool writeRegistersAbsolute(Address address, const std::vector<RegisterValue>& data) = 0;

            virtual bool readRegisters(Address startAddress, Quantity quantity, std::vector<RegisterValue>& data) = 0;

            virtual bool writeRegisters(Address startAddress, const std::vector<RegisterValue>& data) = 0;

            virtual bool readCoils(Address startAddress, Quantity quantity, std::vector<bool>& data) = 0;

            virtual bool writeCoil(Address address, bool state) = 0;

            virtual bool writeCoils(Address startAddress, const std::vector<bool>& states) = 0;
        };
	
	}
}