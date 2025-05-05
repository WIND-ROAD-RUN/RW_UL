#ifndef DSL_PRIORITYQUEUE_H
#define DSL_PRIORITYQUEUE_H

#include "dsl_core.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <functional>
#include<mutex>

namespace rw
{
	namespace dsl
	{
		/**
		 * @brief: Priority Queue Interface
		 *
		 * @template variable:
		 *   - T: The type of the elements in the priority queue
		 *   - Priority: The type of the priority of the elements in the priority queue
		 *
		 * @details: Detailed description of the template class
		 *
		 */
		template <class T, class Priority = size_t>
		class IPriorityQueue
		{
		public:
			/**
			 *@Parameters:
			 *  -a: The first element to be compared
			 *  -b: The second element to be compared
			 *@Methods:
			 * Compare two elements to determine whether they are equal
			 *@Returns: bool
			 * If the two elements are equal, return true, otherwise return false
			 *@Throws:
			 *
			 */
			using CompareNodeEqual = std::function<bool(const T&, const T&)>;

			/**
			 *@Parameters:
			 *  -a: The first priority to be compared
			 *  -b: The second priority to be compared
			 *@Methods:
			 * Compare two priorities to determine the priority of the two elements
			 *@Returns: bool
			 * If the first priority is higher than the second priority, return true, otherwise return false
			 *@Throws:
			 *
			 */
			using CompareNodePriority = std::function<bool(const Priority&, const Priority&)>;

		public:
			/**
			 *@Parameters:
			 *  -CompareNodeEqual: The function to compare two elements
			 *  -CompareNodePriority: The function to compare two priorities
			 *
			 *@Methods:
			 *  Constructor
			 *@Returns: void
			 *
			 *@Throws:
			 *
			 */
			IPriorityQueue(
				CompareNodeEqual compareNodeEqual,
				CompareNodePriority compareNodePriority)
				:_compareNodeEqual(compareNodeEqual), _compareNodePriority(compareNodePriority) {
			}

			IPriorityQueue() = default;

			virtual ~IPriorityQueue() = default;
		protected:
			CompareNodeEqual _compareNodeEqual{ [](const T& a,const T& b) {
					return a == b;
				} };
			CompareNodePriority _compareNodePriority{ [](const Priority& a, const Priority& b) {
					return a < b;
				} };

		public:
			/**
		   *@Parameters:
		   *  void
		   *@Methods:
		   *  Get the top element of the priority queue and remove it
		   *@Returns: template variable T
		   *  The top element of the priority queue
		   *@Throws:
		   *  1. std::runtime_error: If the priority queue is empty
		   */
			virtual T top() = 0;

			/**
			*@Parameters:
			*  void
			*@Methods:
			*  Get the top element of the priority queue without removing it
			*@Returns: template variable T
			*  The top element of the priority queue
			*@Throws:
			*  1. std::runtime_error: If the priority queue is empty
			*/
			virtual T peek() = 0;

			/**
			 *@Parameters:
			 *  1. T element: The element to be inserted
			 *  2. size_t priority: The priority of the element
			 *@Methods:
			 *  Insert an element into the priority queue with a priority
			 *@Returns: void
			 *@Throws: void
			 *@Warning:
			 *  1.If the insert element 's value is the same as the existing element, the new element will be inserted as a new element
			 *  2.If some elements have the same priority, the order of the elements is not guaranteed
			 */
			virtual void insert(T element, Priority priority) = 0;

			/**
			 *@Parameters:
			 *  1. T element: The element to be removed
			 *@Methods:
			 *  Remove an element from the priority queue
			 *@Returns: void
			 *@Throws: void
			 *@Warning:
			 * 1. If the element is not in the priority queue, the method will do nothing
			 * 2. If some elements in the priority queue have the same value,the element to be removed is not guaranteed
			 */
			virtual void remove(T element) = 0;

			/**
			*@Parameters:
			*  1. T element: The element to be updated
			*  2. size_t priority: The new priority of the element
			*@Methods:
			*  Update the priority of an element in the priority queue
			*@Returns: void
			*@Throws: void
			*@Warning:
			* 1. If the element is not in the priority queue, the method will do nothing
			* 2. If some elements in the priority queue have the same value,the element to be updated is not guaranteed
			*/
			virtual void update(T element, Priority priority) = 0;

			/**
			*@Parameters:
			*  void
			*@Methods:
			*  Get the size of the priority queue
			*@Returns: size_t
			*  The size of the priority queue
			*@Throws: void
			*/
			virtual size_t size() = 0;

			/**
			 *@Parameters:
			 *  void
			 *@Methods:
			 *  Clear the priority queue
			 *@Returns: void
			 *
			 *@Throws:
			 *
			 */
			virtual void clear() = 0;
		public:

			/**
			 *@Parameters:
			 *  compareNodeEqual: The function to compare two elements
			 *@Methods:
			 *  Set the function to compare two elements, if the priority queue is not empty, throw an exception
			 *@Returns: void
			 *
			 *@Throws:
			 *  -std::runtime_error: If the priority queue is not empty
			 */
			void setCompareNodeEqual(CompareNodeEqual compareNodeEqual)
			{
				if (this->size()) {
					throw std::runtime_error("Can not set compareNodeEqual");
				}
				this->_compareNodeEqual = compareNodeEqual;
			}

			/**
			 *@Parameters:
			 *  -compareNodePriority: The function to compare two priorities
			 *@Methods:
			 *  Set the function to compare two priorities, if the priority queue is not empty, throw an exception
			 *@Returns: void
			 *
			 *@Throws:
			 *  -std::runtime_error: If the priority queue is not empty
			 */
			void setCompareNodePriority(CompareNodePriority compareNodePriority)
			{
				if (this->size()) {
					throw std::runtime_error("Can not set compareNodePriority");
				}
				this->_compareNodePriority = compareNodePriority;
			}
		};

		template <class T, class Priority = size_t>
		class DHeap : public IPriorityQueue<T, Priority>
		{
		public:
			/**
			 *@Parameters:
			 *  -d: The number of children of each node
			 *     note: According to test,when d=4 or d=5, the performance of the heap is better
			 *@Methods:
			 *  Constructor
			 *@Returns: void
			 *
			 *@Throws:
			 *
			 */
			DHeap(size_t d = 4) :_d(d) {}

			/**
			*@Parameters:
			*  -compareNodeEqual: The function to compare two elements
			*  -compareNodePriority: The function to compare two priorities
			*  -d: The number of children of each node
			*@Methods:
			*  Constructor
			*@Returns: void
			*
			*@Throws:
			*
			*/
			DHeap(std::function<bool(const T&, const T&)> compareNodeEqual,
				std::function<bool(const Priority&, const Priority&)> compareNodePriority,
				size_t d = 4) :_d(d) {
				this->_compareNodeEqual = compareNodeEqual;
				this->_compareNodePriority = compareNodePriority;
			}
			~DHeap() = default;

		public:
			T top() override {
				if (this->_heap_array.empty()) {
					throw std::runtime_error("Heap is empty");
				}
				T top_element = this->_heap_array.front().first;
				std::swap(this->_heap_array.front(), this->_heap_array.back());
				this->_heap_array.pop_back();
				if (!this->_heap_array.empty()) {
					push_down(0);
				}
				return top_element;
			}

			T peek() override {
				if (this->_heap_array.empty()) {
					throw std::runtime_error("Heap is empty");
				}
				return this->_heap_array.front().first;
			}
			void insert(T element, Priority priority) override {
				this->_heap_array.emplace_back(element, priority);
				bubble_up(this->_heap_array.size() - 1);
			}
			void remove(T element) override {
				auto it = std::find_if(this->_heap_array.begin(), this->_heap_array.end(),
					[element, this](const std::pair<T, Priority>& p) { return  this->_compareNodeEqual(p.first, element); });
				if (it != this->_heap_array.end()) {
					std::swap(*it, this->_heap_array.back());
					this->_heap_array.pop_back();
					if (it != this->_heap_array.end()) {
						bubble_up(std::distance(this->_heap_array.begin(), it));
						push_down(std::distance(this->_heap_array.begin(), it));
					}
				}
			}
			void update(T element, Priority priority) override {
				auto it = std::find_if(this->_heap_array.begin(), this->_heap_array.end(),
					[element, this](const std::pair<T, Priority>& p) { return  this->_compareNodeEqual(p.first, element); });
				if (it != this->_heap_array.end()) {
					it->second = priority;
					bubble_up(std::distance(this->_heap_array.begin(), it));
					push_down(std::distance(this->_heap_array.begin(), it));
				}
			}
			size_t size() override {
				return this->_heap_array.size();
			}

			void clear() override {
				this->_heap_array.clear();
			}

		private:
			void bubble_up(size_t index) {
				while (index > 0) {
					size_t p = parent(index);
					if (compare(index, p)) {
						std::swap(this->_heap_array[index], this->_heap_array[p]);
						index = p;
					}
					else {
						break;
					}
				}
			}
			void push_down(size_t index) {
				size_t child_index = child(index, 1);
				while (child_index < this->_heap_array.size()) {
					size_t min_index = index;
					for (size_t i = 0; i < _d; i++) {
						if (child_index + i < this->_heap_array.size() && compare(child_index + i, min_index)) {
							min_index = child_index + i;
						}
					}
					if (min_index != index) {
						std::swap(this->_heap_array[index], this->_heap_array[min_index]);
						index = min_index;
						child_index = child(index, 1);
					}
					else {
						break;
					}
				}
			}
			size_t parent(size_t index) const {
				return (index - 1) / _d;
			}
			size_t child(size_t index, size_t k) const {
				return index * _d + k;
			}
			bool compare(size_t i, size_t j) {
				return this->_compareNodePriority(this->_heap_array[i].second, this->_heap_array[j].second);
			}

		private:
			size_t _d;
			std::vector<std::pair<T, Priority>> _heap_array;
		};

		template <class T, class Priority = size_t>
		class ThreadSafeDHeap : public IPriorityQueue<T, Priority>
		{
		public:
			ThreadSafeDHeap(size_t d = 4) : _d(d) {}

			ThreadSafeDHeap(std::function<bool(const T&, const T&)> compareNodeEqual,
				std::function<bool(const Priority&, const Priority&)> compareNodePriority,
				size_t d = 4)
				: _d(d)
			{
				this->_compareNodeEqual = compareNodeEqual;
				this->_compareNodePriority = compareNodePriority;
			}

			~ThreadSafeDHeap() = default;

		public:
			T top() override {
				std::lock_guard<std::mutex> lock(_mutex);
				if (this->_heap_array.empty()) {
					return 0;
				}
				T top_element = this->_heap_array.front().first;
				std::swap(this->_heap_array.front(), this->_heap_array.back());
				this->_heap_array.pop_back();
				if (!this->_heap_array.empty()) {
					push_down(0);
				}
				return top_element;
			}

			T peek() override {
				std::lock_guard<std::mutex> lock(_mutex);
				if (this->_heap_array.empty()) {
					return 0;
				}
				return this->_heap_array.front().first;
			}

			void insert(T element, Priority priority) override {
				std::lock_guard<std::mutex> lock(_mutex);
				this->_heap_array.emplace_back(element, priority);
				bubble_up(this->_heap_array.size() - 1);
			}

			void remove(T element) override {
				std::lock_guard<std::mutex> lock(_mutex);
				auto it = std::find_if(this->_heap_array.begin(), this->_heap_array.end(),
					[element, this](const std::pair<T, Priority>& p) {
						return this->_compareNodeEqual(p.first, element);
					});
				if (it != this->_heap_array.end()) {
					std::swap(*it, this->_heap_array.back());
					this->_heap_array.pop_back();
					if (it != this->_heap_array.end()) {
						bubble_up(std::distance(this->_heap_array.begin(), it));
						push_down(std::distance(this->_heap_array.begin(), it));
					}
				}
			}

			void update(T element, Priority priority) override {
				std::lock_guard<std::mutex> lock(_mutex);
				auto it = std::find_if(this->_heap_array.begin(), this->_heap_array.end(),
					[element, this](const std::pair<T, Priority>& p) {
						return this->_compareNodeEqual(p.first, element);
					});
				if (it != this->_heap_array.end()) {
					it->second = priority;
					bubble_up(std::distance(this->_heap_array.begin(), it));
					push_down(std::distance(this->_heap_array.begin(), it));
				}
			}

			size_t size() override {
				std::lock_guard<std::mutex> lock(_mutex);
				return this->_heap_array.size();
			}

			void clear() override {
				std::lock_guard<std::mutex> lock(_mutex);
				this->_heap_array.clear();
			}

		private:
			void bubble_up(size_t index) {
				while (index > 0) {
					size_t p = parent(index);
					if (compare(index, p)) {
						std::swap(this->_heap_array[index], this->_heap_array[p]);
						index = p;
					}
					else {
						break;
					}
				}
			}

			void push_down(size_t index) {
				size_t child_index = child(index, 1);
				while (child_index < this->_heap_array.size()) {
					size_t min_index = index;
					for (size_t i = 0; i < _d; i++) {
						if (child_index + i < this->_heap_array.size() && compare(child_index + i, min_index)) {
							min_index = child_index + i;
						}
					}
					if (min_index != index) {
						std::swap(this->_heap_array[index], this->_heap_array[min_index]);
						index = min_index;
						child_index = child(index, 1);
					}
					else {
						break;
					}
				}
			}

			size_t parent(size_t index) const {
				return (index - 1) / _d;
			}

			size_t child(size_t index, size_t k) const {
				return index * _d + k;
			}

			bool compare(size_t i, size_t j) {
				return this->_compareNodePriority(this->_heap_array[i].second, this->_heap_array[j].second);
			}

		private:
			size_t _d;
			std::vector<std::pair<T, Priority>> _heap_array;
			std::mutex _mutex;
		};
	}
}

#endif //DSL_PRIORITY_QUEUE_H