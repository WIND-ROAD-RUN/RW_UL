#pragma once

#include "dsl_Cache.hpp"
#include "dsl_core.hpp"
#include <unordered_map>
#include <queue>
#include <memory>
#include <mutex>
#include <optional>
#include <chrono>
#include "Utilty.hpp"

namespace rw {
	namespace dsl {
		// 自定义哈希函数
		struct TimePointHash {
			std::size_t operator()(const std::chrono::system_clock::time_point& tp) const noexcept {
				return std::hash<std::int64_t>()(tp.time_since_epoch().count());
			}
		};

		template <typename Key, typename Value>
		class ThreadSafeFIFO final : public ICache<Key, Value> {
		public:
			using CacheNode = typename ICache<Key, Value>::CacheNode;

		public:
			explicit ThreadSafeFIFO(size_t capacity) : ICache<Key, Value>(capacity) {}

			std::optional<Value> get(const Key& key) override {
				std::lock_guard<std::mutex> lock(_mutex);
				auto it = _cache.find(key);
				if (it == _cache.end()) {
					return std::nullopt;
				}
				return it->second;
			}

			bool set(const Key& key, const Value& value) override {
				std::lock_guard<std::mutex> lock(_mutex);
				auto it = _cache.find(key);
				if (it != _cache.end()) {
					it->second = value;
					return false;
				}
				if (_cache.size() >= this->_capacity) {
					_cache.erase(_fifo.front());
					_fifo.pop();
				}
				_cache.insert({ key, value });
				_fifo.push(key);
				return true;
			}

			[[nodiscard]] size_t size() const override {
				std::lock_guard<std::mutex> lock(_mutex);
				return _cache.size();
			}

			void clear() override {
				std::lock_guard<std::mutex> lock(_mutex);
				_cache.clear();
				while (!_fifo.empty()) {
					_fifo.pop();
				}
			}

			bool resizeCapacity(size_t capacity) override {
				std::lock_guard<std::mutex> lock(_mutex);
				if (capacity < this->_capacity) {
					while (_cache.size() > capacity) {
						_cache.erase(_fifo.front());
						_fifo.pop();
					}
				}
				this->_capacity = capacity;
				return true;
			}

		private:
			mutable std::mutex _mutex;
			std::unordered_map<Key, Value, TimePointHash> _cache; // 使用自定义哈希函数
			std::queue<Key> _fifo;
		};

		// 特化 Time 和 bool 的版本
		template <>
		class ThreadSafeFIFO<std::chrono::system_clock::time_point, bool> final : public ICache<std::chrono::system_clock::time_point, bool> {
		public:
			using CacheNode = typename ICache<std::chrono::system_clock::time_point, bool>::CacheNode;
			using Time = std::chrono::system_clock::time_point;

			explicit ThreadSafeFIFO(size_t capacity) : ICache<Time, bool>(capacity) {}

			std::optional<bool> get(const Time& key) override {
				std::lock_guard<std::mutex> lock(_mutex);
				auto it = _cache.find(key);
				if (it == _cache.end()) {
					return std::nullopt;
				}
				return it->second;
			}

			bool set(const Time& key, const bool& value) override {
				std::lock_guard<std::mutex> lock(_mutex);
				auto it = _cache.find(key);
				if (it != _cache.end()) {
					it->second = value;
					return false;
				}
				if (_cache.size() >= this->_capacity) {
					_cache.erase(_fifo.front());
					_fifo.pop();
				}
				_cache.insert({ key, value });
				_fifo.push(key);
				return true;
			}

			[[nodiscard]] size_t size() const override {
				std::lock_guard<std::mutex> lock(_mutex);
				return _cache.size();
			}

			void clear() override {
				std::lock_guard<std::mutex> lock(_mutex);
				_cache.clear();
				while (!_fifo.empty()) {
					_fifo.pop();
				}
			}

			bool resizeCapacity(size_t capacity) override {
				std::lock_guard<std::mutex> lock(_mutex);
				if (capacity < this->_capacity) {
					while (_cache.size() > capacity) {
						_cache.erase(_fifo.front());
						_fifo.pop();
					}
				}
				this->_capacity = capacity;
				return true;
			}

		private:
			mutable std::mutex _mutex;
			std::unordered_map<Time, bool, TimePointHash> _cache; // 使用自定义哈希函数
			std::queue<Time> _fifo;
		};
	} // namespace dsl
} // namespace rw