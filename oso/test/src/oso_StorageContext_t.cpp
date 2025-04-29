#include"oso_StorageContext_t.hpp"

namespace oso_StorageContext
{
    TEST_P(StorageContextTest, apiSaveByPath) {
        auto& [type] = GetParam();
        initestObj(type);
        try
        {
            _testObj->save(_sampleAssembly, _testFileName);
        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

        SUCCEED();

    }

    TEST_P(StorageContextTest, apiSaveByPathString) {
        auto& [type] = GetParam();
        initestObj(type);
        try
        {
            _testObj->save(_sampleAssembly, _testFileName.c_str());
        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

        SUCCEED();

    }

    TEST_P(StorageContextTest, apiLoadByPath) {
        auto& [type] = GetParam();
        initestObj(type);
        try
        {
            _testObj->save(_sampleAssembly, _testFileName);
        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

        try
        {
            auto loadedAssembly = _testObj->load(_testFileName);
            ASSERT_TRUE(loadedAssembly != nullptr);
            ASSERT_EQ(loadedAssembly->getName(), _sampleAssembly.getName());
            ASSERT_EQ(loadedAssembly->getItems().size(), _sampleAssembly.getItems().size());
            ASSERT_EQ((*loadedAssembly.get()), _sampleAssembly);

        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

    }

    TEST_P(StorageContextTest, apiLoadByPathString) {
        auto& [type] = GetParam();
        initestObj(type);
        try
        {
            _testObj->save(_sampleAssembly, _testFileName.c_str());
        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

        try
        {
            auto loadedAssembly = _testObj->load(_testFileName);
            ASSERT_TRUE(loadedAssembly != nullptr);
            ASSERT_EQ(loadedAssembly->getName(), _sampleAssembly.getName());
            ASSERT_EQ(loadedAssembly->getItems().size(), _sampleAssembly.getItems().size());
            ASSERT_EQ((*loadedAssembly.get()), _sampleAssembly);
        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

    }

    TEST_P(StorageContextTest, apiGetFormatString) {
        auto& [type] = GetParam();
        initestObj(type);
        try
        {
            _testObj->save(_sampleAssembly, _testFileName);
        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

        try
        {
            auto loadedAssembly = _testObj->load(_testFileName);
            ASSERT_TRUE(loadedAssembly != nullptr);
            ASSERT_EQ(loadedAssembly->getName(), _sampleAssembly.getName());
            ASSERT_EQ(loadedAssembly->getItems().size(), _sampleAssembly.getItems().size());
            ASSERT_EQ(_testObj->getFormatString(*loadedAssembly), _testObj->getFormatString(_sampleAssembly));
        }
        catch (...)
        {
            FAIL() << "Exception thrown";
        }

    }

}
