#include "hoem_ModbusDeviceScheduler.hpp"

namespace rw
{
	namespace hoem
	{
		ModbusDeviceScheduler::ModbusDeviceScheduler(std::shared_ptr<ModbusDevice> device)
			: _device(std::move(device))
		{
			_running.store(true, std::memory_order_release);
			_worker = std::thread([this]() { workerLoop(); });
		}

		ModbusDeviceScheduler::~ModbusDeviceScheduler() {
			stop();
		}

		void ModbusDeviceScheduler::stop() {
			bool expected = true;
			if (_running.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
				_cv.notify_all();
				if (_worker.joinable())
					_worker.join();

				std::vector<Request> leftovers;
				size_t removed = 0;
				{
					std::lock_guard<std::mutex> lk(_mtx);
					while (!_queue.empty()) {
						leftovers.emplace_back(std::move(const_cast<Request&>(_queue.top())));
						_queue.pop();
						++removed;
					}
					if (removed > 0) {
						_pending -= removed;
					}
				}
				for (auto& r : leftovers) {
					if (r.job)
						r.job->cancelShutdown();
				}
				if (removed > 0) {
					_cv.notify_all();
				}
			}
		}

		void ModbusDeviceScheduler::wait() {
			std::unique_lock<std::mutex> lk(_mtx);
			_cv.wait(lk, [this]() { return _pending == 0; });
		}

		size_t ModbusDeviceScheduler::getPendingCount() const {
			std::lock_guard<std::mutex> lk(_mtx);
			return _pending;
		}

		void ModbusDeviceScheduler::workerLoop() {
			while (_running.load(std::memory_order_acquire)) {
				Request req{};
				{
					std::unique_lock<std::mutex> lk(_mtx);
					_cv.wait(lk, [this]() {
						return !_running.load(std::memory_order_acquire) || !_queue.empty();
						});
					if (!_running.load(std::memory_order_acquire))
						break;
					if (_queue.empty())
						continue;

					req = std::move(const_cast<Request&>(_queue.top()));
					_queue.pop();
				}

				bool done = false;
				if (Clock::now() > req.deadline) {
					if (req.job)
						req.job->timeout();
					done = true;
				}
				else {
					try {
						if (req.job)
							req.job->run(*_device);
					}
					catch (...) {
					}
					done = true;
				}

				if (done) {
					std::lock_guard<std::mutex> lk(_mtx);
					if (_pending > 0)
						--_pending;
					if (_pending == 0)
						_cv.notify_all();
				}
			}
		}

		std::future<std::vector<UInt16>> ModbusDeviceScheduler::readRegistersAsync(
			Address16 startAddress, Quantity quantity, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, quantity](ModbusDevice& dev) {
				std::vector<UInt16> data;
				if (!dev.readRegisters(startAddress, quantity, data))
					throw std::runtime_error("readRegisters failed");
				return data;
				});
		}

		std::future<UInt32> ModbusDeviceScheduler::readRegister32Async(
			Address16 startAddress, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, byteOrder](ModbusDevice& dev) {
				UInt32 v{};
				if (!dev.readRegister(startAddress, v, byteOrder))
					throw std::runtime_error("readRegister32 failed");
				return v;
				});
		}

		std::future<std::vector<UInt32>> ModbusDeviceScheduler::readRegisters32Async(
			Address16 startAddress, size_t count, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, count, byteOrder](ModbusDevice& dev) {
				std::vector<UInt32> data(count);
				if (!dev.readRegisters(startAddress, data, byteOrder))
					throw std::runtime_error("readRegisters32 failed");
				return data;
				});
		}

		std::future<std::vector<bool>> ModbusDeviceScheduler::readCoilsAsync(
			Address16 startAddress, Quantity quantity, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, quantity](ModbusDevice& dev) {
				std::vector<bool> data;
				if (!dev.readCoils(startAddress, quantity, data))
					throw std::runtime_error("readCoils failed");
				return data;
				});
		}

		std::future<bool> ModbusDeviceScheduler::readCoilAsync(Address16 startAddress, int prio,
			std::optional<std::chrono::milliseconds> timeout)
		{
			return invokeAsync(prio, timeout, [startAddress](ModbusDevice& dev) {
				bool state{};
				if (!dev.readCoil(startAddress, state))
					throw std::runtime_error("readCoil failed");
				return state;
				});
		}

		std::future<float> ModbusDeviceScheduler::readRegisterFloatAsync(
			Address16 startAddress, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, byteOrder](ModbusDevice& dev) {
				float v{};
				if (!dev.readRegister(startAddress, v, byteOrder))
					throw std::runtime_error("readRegister(float) failed");
				return v;
				});
		}

		std::future<std::vector<float>> ModbusDeviceScheduler::readRegistersFloatAsync(
			Address16 startAddress, size_t count, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, count, byteOrder](ModbusDevice& dev) {
				std::vector<float> data(count);
				if (!dev.readRegisters(startAddress, data, byteOrder))
					throw std::runtime_error("readRegisters(float) failed");
				return data;
				});
		}

		std::future<std::vector<UInt16>> ModbusDeviceScheduler::readRegistersAbsoluteAsync(
			Address16 address, Quantity quantity, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [address, quantity](ModbusDevice& dev) {
				std::vector<UInt16> data;
				if (!dev.readRegistersAbsolute(address, quantity, data))
					throw std::runtime_error("readRegistersAbsolute failed");
				return data;
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeRegistersAsync(
			Address16 startAddress, std::vector<UInt16> data, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, data = std::move(data)](ModbusDevice& dev) mutable {
				return dev.writeRegisters(startAddress, data);
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeRegister32Async(
			Address16 startAddress, UInt32 value, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, value, byteOrder](ModbusDevice& dev) {
				return dev.writeRegister(startAddress, value, byteOrder);
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeRegisters32Async(
			Address16 startAddress, std::vector<UInt32> data, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, data = std::move(data), byteOrder](ModbusDevice& dev) mutable {
				return dev.writeRegisters(startAddress, data, byteOrder);
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeCoilAsync(
			Address16 address, bool state, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [address, state](ModbusDevice& dev) {
				return dev.writeCoil(address, state);
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeCoilsAsync(
			Address16 startAddress, std::vector<bool> states, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, states = std::move(states)](ModbusDevice& dev) mutable {
				return dev.writeCoils(startAddress, states);
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeRegistersAbsoluteAsync(
			Address16 address, std::vector<UInt16> data, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [address, data = std::move(data)](ModbusDevice& dev) mutable {
				return dev.writeRegistersAbsolute(address, data);
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeRegisterFloatAsync(
			Address16 startAddress, float value, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, value, byteOrder](ModbusDevice& dev) {
				return dev.writeRegister(startAddress, value, byteOrder);
				});
		}

		std::future<bool> ModbusDeviceScheduler::writeRegistersFloatAsync(
			Address16 startAddress, std::vector<float> data, Endianness byteOrder, int prio, std::optional<std::chrono::milliseconds> timeout) {
			return invokeAsync(prio, timeout, [startAddress, data = std::move(data), byteOrder](ModbusDevice& dev) mutable {
				return dev.writeRegisters(startAddress, data, byteOrder);
				});
		}
	}
}