#include"oso_StorageContext_t.hpp"

namespace oso_StorageContext
{
	TEST_P(StorageContextTest, apiSaveByPath) {
		auto& [type] = GetParam();
		initestObj(type);
		ASSERT_TRUE(_testObj->save(_sampleAssembly, _testFileName));
	}

	TEST_P(StorageContextTest, apiSaveByPathString) {
		auto& [type] = GetParam();
		initestObj(type);
		ASSERT_TRUE(_testObj->save(_sampleAssembly, _testFileName.c_str()));
	}

	TEST_P(StorageContextTest, apiLoadByPath) {
		auto& [type] = GetParam();
		initestObj(type);
		ASSERT_TRUE(_testObj->save(_sampleAssembly, _testFileName));

		auto loadedAssembly = _testObj->load(_testFileName);
		ASSERT_TRUE(loadedAssembly != nullptr);
		ASSERT_EQ(loadedAssembly->getName(), _sampleAssembly.getName());
		ASSERT_EQ(loadedAssembly->getItems().size(), _sampleAssembly.getItems().size());
		ASSERT_EQ((*loadedAssembly.get()), _sampleAssembly);
	}

	TEST_P(StorageContextTest, apiLoadByPathString) {
		auto& [type] = GetParam();
		initestObj(type);
		ASSERT_TRUE(_testObj->save(_sampleAssembly, _testFileName.c_str()));

		auto loadedAssembly = _testObj->load(_testFileName);
		ASSERT_TRUE(loadedAssembly != nullptr);
		ASSERT_EQ(loadedAssembly->getName(), _sampleAssembly.getName());
		ASSERT_EQ(loadedAssembly->getItems().size(), _sampleAssembly.getItems().size());
		ASSERT_EQ((*loadedAssembly.get()), _sampleAssembly);
	}

	TEST_P(StorageContextTest, apiGetFormatString) {
		auto& [type] = GetParam();
		initestObj(type);
		ASSERT_TRUE(_testObj->save(_sampleAssembly, _testFileName));

		auto loadedAssembly = _testObj->load(_testFileName);
		ASSERT_TRUE(loadedAssembly != nullptr);
		ASSERT_EQ(loadedAssembly->getName(), _sampleAssembly.getName());
		ASSERT_EQ(loadedAssembly->getItems().size(), _sampleAssembly.getItems().size());
		ASSERT_EQ(_testObj->getFormatString(*loadedAssembly), _testObj->getFormatString(_sampleAssembly));
	}
}