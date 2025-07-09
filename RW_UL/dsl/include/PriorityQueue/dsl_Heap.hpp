#pragma once
#include <vector>
#include <shared_mutex>
#include <algorithm>
#include <optional>


namespace rw {
    namespace dsl {
        enum class HeapType {
            MinHeap,
            MaxHeap
        };

        template <typename T, HeapType Type = HeapType::MinHeap>
        class ThreadSafeHeap {
        public:
            // 构造函数，允许预设容量
            explicit ThreadSafeHeap(size_t initialCapacity = 50) {
                heap_.reserve(initialCapacity);
            }

            // 插入元素
            void push(T value) {
                std::unique_lock<std::shared_mutex> lock(mutex_);
                heap_.push_back(value);
                if constexpr (Type == HeapType::MinHeap) {
                    std::push_heap(heap_.begin(), heap_.end(), std::greater<>()); // 最小堆
                }
                else {
                    std::push_heap(heap_.begin(), heap_.end(), std::less<>()); // 最大堆
                }
            }

            // 获取堆顶值（无锁读取）
            bool tryGetTop(T& topValue) {
                std::shared_lock<std::shared_mutex> lock(mutex_);
                if (heap_.empty()) {
                    return false; // 堆为空
                }
                topValue = heap_.front(); // 堆顶即为最值
                return true;
            }

            // 删除堆顶值
            bool tryPopTop(T& topValue) {
                std::unique_lock<std::shared_mutex> lock(mutex_);
                if (heap_.empty()) {
                    return false; // 堆为空
                }
                topValue = heap_.front(); // 堆顶即为最值
                if constexpr (Type == HeapType::MinHeap) {
                    std::pop_heap(heap_.begin(), heap_.end(), std::greater<>()); // 最小堆
                }
                else {
                    std::pop_heap(heap_.begin(), heap_.end(), std::less<>()); // 最大堆
                }
                heap_.pop_back(); // 删除末尾元素
                return true;
            }

            size_t size() const {
                std::shared_lock<std::shared_mutex> lock(mutex_);
                return heap_.size();
            }

        private:
            std::vector<T> heap_; 
            mutable std::shared_mutex mutex_; 
        };
    }

}

