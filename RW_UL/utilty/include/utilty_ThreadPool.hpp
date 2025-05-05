#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <stdexcept>
#include <type_traits>

//TODO:使用优先队列实现线程池
namespace rw {

    /**
     * @brief ThreadPool is a simple thread pool implementation
     *
     * This class provides a simple thread pool implementation that allows you to add tasks 
     *to the thread pool and have them executed by the worker threads.
     *
     * @warning
     *  1.NotThreadSafe : The ThreadPool class is not thread-safe. You should not access the 
     * ThreadPool object from multiple threads at the same time.
     *  2.NotHandleException : The ThreadPool class does not handle exceptions thrown by the 
     * tasks. If a task throws an exception, the exception will be propagated to the caller 
     * of the get() method.
     *  Suggestions: You can use a try-catch block inside the task to catch the exception and 
     * handle it appropriately.
     * 
     * @example
     *      //create a thread pool with the number of threads equal to the number of cores in the CPU
     *      ThreadPool pool(std::thread::hardware_concurrency());
     *      auto result = pool.enqueue([] { return 42; });//add a task to the thread pool
     *      std::cout << result.get() << std::endl;//get the result of the task
     */
    class ThreadPool {
    public:

        /**
         *@Parameters:
         *  - numThreads: The number of threads in the thread pool,it usually equals to the 
         *number of cores in the CPU
         *@Methods:
         *  constructor: Create a thread pool with the specified number of threads
         *@Returns: void
         *
         *@Throws:
         *
         */
        explicit ThreadPool(size_t numThreads);

        /**
         * @brief Destructor
         *
         * The destructor will stop the worker threads and join them
         */
        ~ThreadPool();

        /**
         *@Parameters:
         *  - f: The function to be executed
         *  - args: The arguments to be passed to the function
         *@Methods:
         *  enqueue: Add a task to the task queue
         *@Returns: std::future<typename std::invoke_result<F, Args...>::type>
         *  return a future object that can be used to get the result of the function
         *@Throws:
         *  
         */
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;

    private:

        /**
         * @brief The worker threads
         *
         * The worker threads will pull tasks from the task queue and execute
         */
        std::vector<std::thread> workers; 

        /**
         * @brief The task queue
         *
         * The task queue is a queue of tasks that need to be executed by the worker threads
         */
        std::queue<std::function<void()>> tasks;

        /**
         * @brief The mutex to protect the task queue
         *
         * The mutex is used to protect the task queue from being accessed by multiple threads 
         *at the same time
         */
        std::mutex queueMutex;

        /**
         * @brief The condition variable to notify the worker threads
         *
         * The condition variable is used to notify the worker threads that there is a new 
         *task in the task queue
         */
        std::condition_variable condition;

        /**
         * @brief A flag to indicate that the thread pool should stop
         *
         * The stop flag is used to indicate that the thread pool should stop and that the 
         *worker threads should exit
         */
        bool stop;
    };


    /*------------------------------------------------------------------------------------
     *                          Inline implementation of ThreadPool
     *-----------------------------------------------------------------------------------*/

    ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
                });
        }
    }

    // 添加任务到线程池
    template<class F, class... Args>
    auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using returnType = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<returnType()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<returnType> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ThreadPool::~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

}