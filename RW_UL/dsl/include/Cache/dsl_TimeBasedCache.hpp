#pragma once

#include <deque>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include<chrono>

namespace rw {
    namespace dsl {
        template <typename Time, typename T>
        class TimeBasedCache {
        public:
            explicit TimeBasedCache(size_t capacity) : _capacity(capacity) {}

            // 插入数据
            void insert(const Time& time, const T& data) {
                std::lock_guard<std::mutex> lock(_mutex);

                // 如果缓存已满，删除最旧的数据
                if (_cache.size() >= _capacity) {
                    _cache.pop_front();
                }

                // 插入新数据
                _cache.emplace_back(time, data);
            }

            std::vector<T> query(const Time& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);

                if (count <= 0) return {};

                // 1. 先筛选方向
                std::vector<std::pair<Time, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first < time) {
                        candidates.push_back(entry);
                    }
                    if (!isBefore && entry.first > time) {
                        candidates.push_back(entry);
                    }
                }

                // 2. 按与 time 的距离排序
                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs(a.first - time) < std::abs(b.first - time);
                    });

                // 3. 取前 count 个
                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                // 4. 按时间排序
                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                // 5. 只返回数据部分
                std::vector<T> result;
                for (const auto& entry : candidates) {
                    result.push_back(entry.second);
                }
                return result;
            }

            std::vector<T> queryWithTime(const Time& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);

                if (count <= 0) return {};

                // 收集所有与 time 的距离
                std::vector<std::pair<Time, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first > time) continue;
                    if (!isBefore && entry.first < time) continue;
                    candidates.push_back(entry);
                }

                // 按距离排序
                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs(a.first - time) < std::abs(b.first - time);
                    });

                // 取前 count 个
                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                // 按时间排序
                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                // 只返回数据部分
                std::vector<T> result;
                for (const auto& entry : candidates) {
                    result.push_back(entry.second);
                }
                std::cout << result.size() << std::endl;
                return result;
            }

            std::unordered_map<Time, T> queryToMap(const Time& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);

                if (count <= 0) return {};

                // 1. 先筛选方向
                std::vector<std::pair<Time, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first < time) {
                        candidates.push_back(entry);
                    }
                    if (!isBefore && entry.first > time) {
                        candidates.push_back(entry);
                    }
                }

                // 2. 按与 time 的距离排序
                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs(a.first - time) < std::abs(b.first - time);
                    });

                // 3. 取前 count 个
                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                // 4. 按时间排序
                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                // 5. 返回 map
                std::unordered_map<Time, T> result;
                for (const auto& entry : candidates) {
                    result[entry.first] = entry.second;
                }
                return result;
            }

            std::unordered_map<Time, T> queryWithTimeToMap(const Time& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);

                if (count <= 0) return {};

                // 收集所有与 time 的距离
                std::vector<std::pair<Time, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first > time) continue;
                    if (!isBefore && entry.first < time) continue;
                    candidates.push_back(entry);
                }

                // 按距离排序
                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs(a.first - time) < std::abs(b.first - time);
                    });

                // 取前 count 个
                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                // 按时间排序
                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                // 返回 map
                std::unordered_map<Time, T> result;
                for (const auto& entry : candidates) {
                    result[entry.first] = entry.second;
                }
                return result;
            }

            // 获取缓存大小
            size_t size() const {
                std::lock_guard<std::mutex> lock(_mutex);
                return _cache.size();
            }

        private:
            mutable std::mutex _mutex; // 保护缓存的线程安全
            size_t _capacity;          // 缓存容量
            std::deque<std::pair<Time, T>> _cache; // 缓存数据，存储时间点和数据的键值对
        };

        template <typename T>
        class TimeBasedCache<std::chrono::system_clock::time_point, T> {
        public:
            explicit TimeBasedCache(size_t capacity) : _capacity(capacity) {}

            void insert(const std::chrono::system_clock::time_point& time, const T& data) {
                std::lock_guard<std::mutex> lock(_mutex);
                if (_cache.size() >= _capacity) {
                    _cache.pop_front();
                }
                _cache.emplace_back(time, data);
            }

            std::vector<T> queryWithTime(const std::chrono::system_clock::time_point& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);

                if (count <= 0) return {};

                // 收集所有与 time 的距离
                std::vector<std::pair<std::chrono::system_clock::time_point, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first > time) continue;
                    if (!isBefore && entry.first < time) continue;
                    candidates.push_back(entry);
                }

                // 按距离排序
                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs((a.first - time).count()) < std::abs((b.first - time).count());
                    });

                // 取前 count 个
                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                // 按时间排序
                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                // 只返回数据部分
                std::vector<T> result;
                for (const auto& entry : candidates) {
                    result.push_back(entry.second);
                }
                return result;
            }

            std::unordered_map<std::chrono::system_clock::time_point, T> queryToMap(const std::chrono::system_clock::time_point& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);

                if (count <= 0) return {};

                // 收集所有与 time 的距离
                std::vector<std::pair<std::chrono::system_clock::time_point, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first < time) {
                        candidates.push_back(entry);
                    }
                    if (!isBefore && entry.first > time) {
                        candidates.push_back(entry);
                    }
                }

                // 按距离排序
                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs((a.first - time).count()) < std::abs((b.first - time).count());
                    });

                // 取前 count 个
                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                // 按时间排序
                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                // 返回 map
                std::unordered_map<std::chrono::system_clock::time_point, T> result;
                for (const auto& entry : candidates) {
                    result[entry.first] = entry.second;
                }
                return result;
            }

            std::unordered_map<std::chrono::system_clock::time_point, T> queryWithTimeToMap(const std::chrono::system_clock::time_point& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);

                if (count <= 0) return {};

                // 收集所有与 time 的距离
                std::vector<std::pair<std::chrono::system_clock::time_point, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first > time) continue;
                    if (!isBefore && entry.first < time) continue;
                    candidates.push_back(entry);
                }

                // 按距离排序
                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs((a.first - time).count()) < std::abs((b.first - time).count());
                    });

                // 取前 count 个
                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                // 按时间排序
                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                // 返回 map
                std::unordered_map<std::chrono::system_clock::time_point, T> result;
                for (const auto& entry : candidates) {
                    result[entry.first] = entry.second;
                }
                return result;
            }

            std::vector<T> query(const std::chrono::system_clock::time_point& time, int count, bool isBefore = true, bool ascending = true) const {
                std::lock_guard<std::mutex> lock(_mutex);
                if (count <= 0) return {};

                std::vector<std::pair<std::chrono::system_clock::time_point, T>> candidates;
                for (const auto& entry : _cache) {
                    if (isBefore && entry.first < time) {
                        candidates.push_back(entry);
                    }
                    if (!isBefore && entry.first > time) {
                        candidates.push_back(entry);
                    }
                }

                std::sort(candidates.begin(), candidates.end(), [&time](const auto& a, const auto& b) {
                    return std::abs((a.first - time).count()) < std::abs((b.first - time).count());
                    });

                if (candidates.size() > static_cast<size_t>(count)) {
                    candidates.resize(count);
                }

                if (ascending) {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first;
                        });
                }
                else {
                    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first;
                        });
                }

                std::vector<T> result;
                for (const auto& entry : candidates) {
                    result.push_back(entry.second);
                }
                return result;
            }

            size_t size() const {
                std::lock_guard<std::mutex> lock(_mutex);
                return _cache.size();
            }

        private:
            mutable std::mutex _mutex;
            size_t _capacity;
            std::deque<std::pair<std::chrono::system_clock::time_point, T>> _cache;
        };
    }
}