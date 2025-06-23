#pragma once
#include <string>
#include <vector>

namespace rw {
	namespace hoei {

		struct CPUInfo
		{
		public:
			// 默认构造函数
			CPUInfo()=default;

			// 拷贝构造函数
			CPUInfo(const CPUInfo& other);

			// 移动构造函数
			CPUInfo(CPUInfo&& other) noexcept;

			// 拷贝赋值运算符
			CPUInfo& operator=(const CPUInfo& other);

			// 移动赋值运算符
			CPUInfo& operator=(CPUInfo&& other) noexcept;
		public:
			void getCurrentContextInfo();
		public:
			std::string cpuModel{};
		public:
			size_t coreCount{};
		public:
			size_t logicCoreCount{};
		public:
			size_t threadCount{};
		public:
			//MHz
			double baseClockSpeed{};
		public:
			struct Topology {
				std::string type; // 类型，例如 "L1 Cache", "Core", "NUMA Node"
				size_t depth;     // 深度
				size_t index;     // 索引
				double size;      // 大小（例如缓存大小，单位 KB）
				std::string description; // 描述信息
			};
			std::vector<Topology> topology;
		};

		class CPUInfoFactory
		{
		public:
			static CPUInfo createCPUInfo();
		public:
			static std::string GetCPUModel();
			static size_t GetCoreCount();
			static size_t GetLogicCount();
			static size_t GetThreadCount();
			static double GetBaseClockSpeed();
			static std::vector<CPUInfo::Topology> GeyTopology();
		};
	
	}

}