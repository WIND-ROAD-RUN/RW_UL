#ifndef DSL_PRIORITYQUEUE_H
#define DSL_PRIORITYQUEUE_H

#include"dsl_IPriorityQueue.hpp"

#include"dsl_DHeap.hpp"

		private:
			size_t _d;
			std::vector<std::pair<T, Priority>> _heap_array;
			std::mutex _mutex;
		};
	}
}

#endif //DSL_PRIORITY_QUEUE_H