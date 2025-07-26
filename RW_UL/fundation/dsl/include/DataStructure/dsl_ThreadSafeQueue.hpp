#pragma once
#include <mutex>
#include <queue>

namespace rw
{
	namespace dsl{
		template<typename T>
		class ThreadSafeQueue
		{
		public:
			ThreadSafeQueue() = default;
			void push(const T& value)
			{
				std::lock_guard<std::mutex> lock(_mutex);
				_queue.push(value);
			}
			bool pop(T& value)
			{
				std::lock_guard<std::mutex> lock(_mutex);
				if (_queue.empty())
					return false;
				value = _queue.front();
				_queue.pop();
				return true;
			}
			bool empty() const
			{
				std::lock_guard<std::mutex> lock(_mutex);
				return _queue.empty();
			}
			size_t size() const
			{
				std::lock_guard<std::mutex> lock(_mutex);
				return _queue.size();
			}

			T& back()
			{
				std::lock_guard<std::mutex> lock(_mutex);
				if (_queue.empty())
					throw std::out_of_range("ThreadSafeQueue::back() called on empty queue");
				return _queue.back();
			}

			const T& back() const
			{
				std::lock_guard<std::mutex> lock(_mutex);
				if (_queue.empty())
					throw std::out_of_range("ThreadSafeQueue::back() called on empty queue");
				return _queue.back();
			}

		private:
			mutable std::mutex _mutex;
			std::queue<T> _queue;
		};
	}
}
