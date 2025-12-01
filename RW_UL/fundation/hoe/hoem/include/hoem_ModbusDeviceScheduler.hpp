#pragma once

#include "hoem_ModbusDevice.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>
#include <functional>
#include <stdexcept>

namespace rw
{
	namespace hoem
	{
		class ModbusDeviceScheduler {
		public:
			using Clock = std::chrono::steady_clock;
			using TimePoint = Clock::time_point;

			explicit ModbusDeviceScheduler(std::shared_ptr<ModbusDevice> device);
			explicit ModbusDeviceScheduler(const ModbusDevice & device)=delete;
			explicit ModbusDeviceScheduler(ModbusDevice&& device) = delete;
			~ModbusDeviceScheduler();

			ModbusDeviceScheduler(const ModbusDeviceScheduler&) = delete;
			ModbusDeviceScheduler& operator=(const ModbusDeviceScheduler&) = delete;

			void stop();

		private:
			template<typename F>
			auto invokeAsync(int prio, std::optional<std::chrono::milliseconds> timeout, F&& fn)
				-> std::future<typename std::invoke_result<F, ModbusDevice&>::type>
			{
				using R = typename std::invoke_result<F, ModbusDevice&>::type;
				auto deadline = timeout.has_value() ? Clock::now() + *timeout : TimePoint::max();
				return enqueueImpl<R>(clampPriority(prio), deadline, std::forward<F>(fn));
			}

			template<typename F>
			auto invokeAsync(F&& fn)
				-> std::future<typename std::invoke_result<F, ModbusDevice&>::type>
			{
				return invokeAsync(3, std::nullopt, std::forward<F>(fn));
			}
		public:
			std::future<std::vector<UInt16>> readRegistersAsync(Address16 startAddress, Quantity quantity,
				int prio = 8, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<UInt32> readRegister32Async(Address16 startAddress, Endianness byteOrder,
				int prio = 8, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<std::vector<UInt32>> readRegisters32Async(Address16 startAddress, size_t count, Endianness byteOrder,
				int prio = 8, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<std::vector<bool>> readCoilsAsync(Address16 startAddress, Quantity quantity,
				int prio = 8, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<float> readRegisterFloatAsync(Address16 startAddress, Endianness byteOrder,
				int prio = 8, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<std::vector<float>> readRegistersFloatAsync(Address16 startAddress, size_t count, Endianness byteOrder,
				int prio = 8, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<std::vector<UInt16>> readRegistersAbsoluteAsync(Address16 address, Quantity quantity,
				int prio = 8, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeRegistersAsync(Address16 startAddress, std::vector<UInt16> data,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeRegister32Async(Address16 startAddress, UInt32 data, Endianness byteOrder,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeRegisters32Async(Address16 startAddress, std::vector<UInt32> data, Endianness byteOrder,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeCoilAsync(Address16 address, bool state,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeCoilsAsync(Address16 startAddress, std::vector<bool> states,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeRegistersAbsoluteAsync(Address16 address, std::vector<UInt16> data,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeRegisterFloatAsync(Address16 startAddress, float value, Endianness byteOrder,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

			std::future<bool> writeRegistersFloatAsync(Address16 startAddress, std::vector<float> data, Endianness byteOrder,
				int prio = 3, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

		private:
			struct IJob {
				virtual ~IJob() = default;
				virtual void run(ModbusDevice& dev) = 0;
				virtual void timeout() = 0;
				virtual void cancelShutdown() = 0;
			};

			template<typename R, typename F>
			struct Job : IJob {
				std::promise<R> p;
				F fn;

				explicit Job(F&& f) : fn(std::forward<F>(f)) {}

				std::future<R> get_future() {
					return p.get_future();
				}

				void run(ModbusDevice& dev) override {
					try {
						if constexpr (std::is_void_v<R>) {
							fn(dev);
							p.set_value();
						}
						else {
							auto r = fn(dev);
							p.set_value(std::move(r));
						}
					}
					catch (...) {
						p.set_exception(std::current_exception());
					}
				}

				void timeout() override {
					try {
						throw std::runtime_error("ModbusDeviceScheduler: deadline exceeded");
					}
					catch (...) {
						p.set_exception(std::current_exception());
					}
				}

				void cancelShutdown() override {
					try {
						throw std::runtime_error("ModbusDeviceScheduler: scheduler stopped");
					}
					catch (...) {
						p.set_exception(std::current_exception());
					}
				}
			};

			struct Request {
				uint64_t seq;
				int prio; // 1~10，越大优先级越高
				TimePoint deadline;
				std::unique_ptr<IJob> job;
			};

			struct RequestCmp {
				bool operator()(Request const& a, Request const& b) const {
					if (a.prio != b.prio)
						return a.prio < b.prio; // 高优先级在前
					if (a.deadline != b.deadline)
						return a.deadline > b.deadline; // 截止时间早者在前
					return a.seq > b.seq; // 先进先出
				}
			};

			template<typename R, typename F>
			std::future<R> enqueueImpl(int prio, TimePoint deadline, F&& fn) {
				auto job = std::make_unique<Job<R, std::decay_t<F>>>(std::forward<F>(fn));
				auto fut = job->get_future();

				{
					std::lock_guard<std::mutex> lk(_mtx);
					_queue.push(Request{ ++_seq, prio, deadline, std::move(job) });
				}
				_cv.notify_one();
				return fut;
			}

			static int clampPriority(int prio) {
				if (prio < 1) return 1;
				if (prio > 10) return 10;
				return prio;
			}

			void workerLoop();

		private:
			std::shared_ptr<ModbusDevice> _device;
			std::priority_queue<Request, std::vector<Request>, RequestCmp> _queue;
			std::thread _worker;
			std::mutex _mtx;
			std::condition_variable _cv;
			std::atomic<bool> _running{ false };
			uint64_t _seq = 0;
		};
	}
}