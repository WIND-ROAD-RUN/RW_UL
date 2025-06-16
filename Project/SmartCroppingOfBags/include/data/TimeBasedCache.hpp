#pragma once

#include <deque>
#include <vector>
#include <chrono>
#include <mutex>
#include <unordered_map>

using Time = std::chrono::system_clock::time_point;

template <typename T>
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

    std::vector<T> query(const Time& time, int count, bool isBefore=true, bool ascending = true) const {
        std::lock_guard<std::mutex> lock(_mutex);

        std::vector<T> result;
        if (count <= 0) {
            return result; // 返回空结果
        }

        if (isBefore) {
            if (ascending) {
                // 查询先于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first < time) {
                        result.push_back(it->second);
                        --count;
                    }
                }
            }
            else {
                // 查询先于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first < time) {
                        result.push_back(it->second);
                        --count;
                    }
                }
            }
        }
        else {
            if (ascending) {
                // 查询晚于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first > time) {
                        result.push_back(it->second);
                        --count;
                    }
                }
            }
            else {
                // 查询晚于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first > time) {
                        result.push_back(it->second);
                        --count;
                    }
                }
            }
        }

        return result;
    }

    std::vector<T> queryWithTime(const Time& time, int count, bool isBefore = true, bool ascending = true) const {
        std::lock_guard<std::mutex> lock(_mutex);

        std::vector<T> result;
        if (count <= 0) {
            return result; // 返回空结果
        }

        // 首先将输入参数 time 作为一个元素加入结果
        result.emplace_back(time);
        --count;

        if (count <= 0) {
            return result; // 如果 count 为 1，直接返回包含 time 的结果
        }

        if (isBefore) {
            if (ascending) {
                // 查询先于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first < time) { // 不包括指定时间点
                        result.emplace_back(it->second);
                        --count;
                    }
                }
            }
            else {
                // 查询先于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first < time) { // 不包括指定时间点
                        result.emplace_back(it->second);
                        --count;
                    }
                }
            }
        }
        else {
            if (ascending) {
                // 查询晚于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first > time) { // 不包括指定时间点
                        result.emplace_back(it->second);
                        --count;
                    }
                }
            }
            else {
                // 查询晚于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first > time) { // 不包括指定时间点
                        result.emplace_back(it->second);
                        --count;
                    }
                }
            }
        }

        return result;
    }

    // 返回哈希表，基于 query 的逻辑
    std::unordered_map<Time, T> queryToMap(const Time& time, int count, bool isBefore = true, bool ascending = true) const {
        std::lock_guard<std::mutex> lock(_mutex);

        std::unordered_map<Time, T> result;
        if (count <= 0) {
            return result; // 返回空结果
        }

        if (isBefore) {
            if (ascending) {
                // 查询先于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first < time) {
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
            else {
                // 查询先于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first < time) {
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
        }
        else {
            if (ascending) {
                // 查询晚于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first > time) {
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
            else {
                // 查询晚于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first > time) {
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
        }

        return result;
    }

    // 返回哈希表，基于 queryWithTime 的逻辑
    std::unordered_map<Time, T> queryWithTimeToMap(const Time& time, int count, bool isBefore = true, bool ascending = true) const {
        std::lock_guard<std::mutex> lock(_mutex);

        std::unordered_map<Time, T> result;
        if (count <= 0) {
            return result; // 返回空结果
        }

        // 检查 _cache 中是否存在输入参数 time
        auto it = std::find_if(_cache.begin(), _cache.end(), [&time](const std::pair<Time, T>& entry) {
            return entry.first == time;
            });

        if (it != _cache.end()) {
            // 如果存在，将其加入结果
            result[time] = it->second;
        }
        else {
            // 如果不存在，使用默认构造的 T 值
            result[time] = T(); // 假设 T 类型有默认构造函数
        }
        --count;

        if (count <= 0) {
            return result; // 如果 count 为 1，直接返回包含 time 的结果
        }

        if (isBefore) {
            if (ascending) {
                // 查询先于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first < time) { // 不包括指定时间点
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
            else {
                // 查询先于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first < time) { // 不包括指定时间点
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
        }
        else {
            if (ascending) {
                // 查询晚于指定时间点的数据，按时间从前往后
                for (auto it = _cache.begin(); it != _cache.end() && count > 0; ++it) {
                    if (it->first > time) { // 不包括指定时间点
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
            else {
                // 查询晚于指定时间点的数据，按时间从后往前
                for (auto it = _cache.rbegin(); it != _cache.rend() && count > 0; ++it) {
                    if (it->first > time) { // 不包括指定时间点
                        result[it->first] = it->second;
                        --count;
                    }
                }
            }
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