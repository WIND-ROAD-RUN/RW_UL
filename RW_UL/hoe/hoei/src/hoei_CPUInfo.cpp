#include"hoei_CPUInfo.hpp"

#include <hwloc.h>

namespace rw
{
	namespace hoei
	{
		CPUInfo::CPUInfo(const CPUInfo& other)
			: cpuModel(other.cpuModel), coreCount(other.coreCount), logicCoreCount(other.logicCoreCount),
			threadCount(other.threadCount), baseClockSpeed(other.baseClockSpeed), topology(other.topology) {
		}

		CPUInfo::CPUInfo(CPUInfo&& other) noexcept
			: cpuModel(std::move(other.cpuModel)), coreCount(other.coreCount), logicCoreCount(other.logicCoreCount),
			threadCount(other.threadCount), baseClockSpeed(other.baseClockSpeed), topology(std::move(other.topology)) {
		}

		CPUInfo& CPUInfo::operator=(const CPUInfo& other)
		{
			if (this != &other) {
				cpuModel = other.cpuModel;
				coreCount = other.coreCount;
				logicCoreCount = other.logicCoreCount;
				threadCount = other.threadCount;
				baseClockSpeed = other.baseClockSpeed;
				topology = other.topology;
			}
			return *this;
		}

		CPUInfo& CPUInfo::operator=(CPUInfo&& other) noexcept
		{
			if (this != &other) {
				cpuModel = std::move(other.cpuModel);
				coreCount = other.coreCount;
				logicCoreCount = other.logicCoreCount;
				threadCount = other.threadCount;
				baseClockSpeed = other.baseClockSpeed;
				topology = std::move(other.topology);
			}
			return *this;
		}

		void CPUInfo::getCurrentContextInfo()
		{
			auto info = CPUInfoFactory::createCPUInfo();
			*this = info;
		}

		CPUInfo CPUInfoFactory::createCPUInfo()
		{
			CPUInfo info;
			info.cpuModel = CPUInfoFactory::GetCPUModel();
			info.coreCount = CPUInfoFactory::GetCoreCount();
			info.logicCoreCount = CPUInfoFactory::GetLogicCount();
			info.threadCount = CPUInfoFactory::GetThreadCount();
			info.baseClockSpeed = CPUInfoFactory::GetBaseClockSpeed();
			info.topology = CPUInfoFactory::GeyTopology();
			return info;
		}

		std::string CPUInfoFactory::GetCPUModel()
		{
			char cpuInfo[0x40] = { 0 };
			int cpuInfoData[4] = { 0 };

			__cpuid(cpuInfoData, 0x80000002);
			memcpy(cpuInfo, cpuInfoData, sizeof(cpuInfoData));

			__cpuid(cpuInfoData, 0x80000003);
			memcpy(cpuInfo + 16, cpuInfoData, sizeof(cpuInfoData));

			__cpuid(cpuInfoData, 0x80000004);
			memcpy(cpuInfo + 32, cpuInfoData, sizeof(cpuInfoData));

			return std::string(cpuInfo);
		}

		size_t CPUInfoFactory::GetCoreCount() {
			// 初始化 hwloc 拓扑
			hwloc_topology_t topology;
			hwloc_topology_init(&topology);
			hwloc_topology_load(topology);

			// 获取物理核心数
			int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
			if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
				hwloc_topology_destroy(topology);
				return 0; // 如果无法获取，返回 0
			}

			size_t coreCount = hwloc_get_nbobjs_by_depth(topology, depth);

			// 销毁 hwloc 拓扑
			hwloc_topology_destroy(topology);

			return coreCount;
		}

		size_t CPUInfoFactory::GetLogicCount() {
			// 初始化 hwloc 拓扑
			hwloc_topology_t topology;
			hwloc_topology_init(&topology);
			hwloc_topology_load(topology);

			// 获取逻辑处理器数
			int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU); // PU 表示处理单元
			if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
				hwloc_topology_destroy(topology);
				return 0; // 如果无法获取，返回 0
			}

			size_t logicCount = hwloc_get_nbobjs_by_depth(topology, depth);

			// 销毁 hwloc 拓扑
			hwloc_topology_destroy(topology);

			return logicCount;
		}

		size_t CPUInfoFactory::GetThreadCount() {
			// 使用 hwloc 获取逻辑处理器数作为线程数
			return GetLogicCount();
		}

		double CPUInfoFactory::GetBaseClockSpeed()
		{
			HKEY hKey;
			DWORD data;
			DWORD dataSize = sizeof(data);

			// 打开注册表键以获取 CPU 基础频率
			if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
				"Hardware\\Description\\System\\CentralProcessor\\0",
				0, KEY_READ, &hKey) == ERROR_SUCCESS) {
				if (RegQueryValueEx(hKey, "~MHz", nullptr, nullptr, reinterpret_cast<LPBYTE>(&data), &dataSize) == ERROR_SUCCESS) {
					RegCloseKey(hKey);
					return static_cast<double>(data);
				}
				RegCloseKey(hKey);
			}
			return 0.0; // 如果无法获取，返回 0.0
		}

		std::vector<CPUInfo::Topology> CPUInfoFactory::GeyTopology()
		{
			std::vector<CPUInfo::Topology> topology;

			// 初始化 hwloc 拓扑
			hwloc_topology_t hwTopology;
			hwloc_topology_init(&hwTopology);
			hwloc_topology_load(hwTopology);

			// 遍历所有对象
			int depth = hwloc_topology_get_depth(hwTopology);
			for (int d = 0; d < depth; ++d) {
				int numObjects = hwloc_get_nbobjs_by_depth(hwTopology, d);
				for (int i = 0; i < numObjects; ++i) {
					hwloc_obj_t obj = hwloc_get_obj_by_depth(hwTopology, d, i);

					CPUInfo::Topology topo;
					topo.type = hwloc_obj_type_string(obj->type); // 获取类型名称
					topo.depth = d;
					topo.index = i;
					topo.size = obj->attr && obj->attr->cache.size ? static_cast<double>(obj->attr->cache.size) / 1024.0 : 0.0; // 缓存大小
					topo.description = obj->name ? obj->name : ""; // 描述信息

					topology.push_back(topo);
				}
			}

			// 销毁 hwloc 拓扑
			hwloc_topology_destroy(hwTopology);

			return topology;
		}
	}
}