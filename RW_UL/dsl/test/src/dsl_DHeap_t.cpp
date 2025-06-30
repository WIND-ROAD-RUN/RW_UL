#include "dsl_pch_t.h"
#include "dsl_DHeap.hpp"

namespace dsl_PriorityQueue {
    // 定义优先级比较函数
    auto compareNodeEqual = [](const int& a, const int& b) -> bool {
        return a == b;
        };

    auto compareNodePriority = [](const size_t& a, const size_t& b) -> bool {
        return a > b; // 优先级较大的元素优先
        };

    TEST(DHeapThreadSafeTest, InsertAndPeekDefault) {
        rw::dsl::DHeapThreadSafe<int> heap;

        heap.insert(10, 1);
        heap.insert(20, 2);
        heap.insert(30, 3);

        EXPECT_EQ(heap.peek(), 10); // Peek should return the element with the highest priority
    }

    TEST(DHeapThreadSafeTest, InsertAndPeek) {
        rw::dsl::DHeapThreadSafe<int> heap(compareNodeEqual, compareNodePriority, 4);

        heap.insert(10, 1);
        heap.insert(20, 2);
        heap.insert(30, 3);

        EXPECT_EQ(heap.peek(), 30); // Peek should return the element with the highest priority
    }

    TEST(DHeapThreadSafeTest, TopAndRemove) {
        rw::dsl::DHeapThreadSafe<int> heap(compareNodeEqual, compareNodePriority, 4);

        heap.insert(10, 1);
        heap.insert(20, 2);
        heap.insert(30, 3);

        EXPECT_EQ(heap.top(), 30); // Top should return and remove the element with the highest priority
        EXPECT_EQ(heap.peek(), 20); // After removing the top, peek should return the next highest priority element

        heap.remove(20);
        EXPECT_EQ(heap.peek(), 10); // After removing 20, peek should return the remaining element
    }

    TEST(DHeapThreadSafeTest, UpdatePriority) {
        rw::dsl::DHeapThreadSafe<int> heap(compareNodeEqual, compareNodePriority, 4);

        heap.insert(10, 1);
        heap.insert(20, 2);
        heap.insert(30, 3);

        heap.update(10, 4); // Update priority of 10 to be the highest
        EXPECT_EQ(heap.peek(), 10); // Peek should now return 10
    }

    TEST(DHeapThreadSafeTest, ClearHeap) {
        rw::dsl::DHeapThreadSafe<int> heap(compareNodeEqual, compareNodePriority, 4);

        heap.insert(10, 1);
        heap.insert(20, 2);
        heap.insert(30, 3);

        heap.clear();
        EXPECT_EQ(heap.size(), 0); // Heap should be empty after clear
        EXPECT_THROW(heap.peek(), std::runtime_error); // Peek should throw an exception
    }

    TEST(DHeapThreadSafeTest, ThreadSafety) {
        rw::dsl::DHeapThreadSafe<int> heap(compareNodeEqual, compareNodePriority, 4);

        auto insertTask = [&heap]() {
            for (int i = 0; i < 100; ++i) {
                heap.insert(i, i);
            }
            };

        auto removeTask = [&heap]() {
            for (int i = 0; i < 50; ++i) {
                if (heap.size() > 0) {
                    heap.top();
                }
            }
            };

        std::thread t1(insertTask);
        std::thread t2(removeTask);

        t1.join();
        t2.join();

        EXPECT_GE(heap.size(), 50); // After concurrent insert and remove, at least 50 elements should remain
    }
}