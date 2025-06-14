#include <gtest/gtest.h>
#include "rqw_HistoricalElementManager_t.hpp"

using namespace rw::rqw;

TEST(HistoricalElementManagerTest, InsertAndRetrieveElement) {
    // 定义测试数据
    using KeyType = int;
    using ImageType = QImage;
    HistoricalElementManager<KeyType, ImageType> manager(10);

    KeyType key = 1;
    QImage image(100, 100, QImage::Format_RGB32);
    ElementInfo<ImageType> elementInfo(image);
    elementInfo.attribute["Author"] = "TestUser";

    // 插入元素
    manager.insertElement(key, elementInfo);

    // 检索元素
    auto retrievedElement = manager.getElement(key);
    ASSERT_TRUE(retrievedElement.has_value());
    EXPECT_EQ(retrievedElement->element.size(), image.size());
    EXPECT_EQ(retrievedElement->attribute["Author"], "TestUser");
}

TEST(HistoricalElementManagerTest, OverwriteElement) {
    using KeyType = int;
    using ImageType = QImage;
    HistoricalElementManager<KeyType, ImageType> manager(10);

    KeyType key = 2;
    QImage image1(100, 100, QImage::Format_RGB32);
    QImage image2(200, 200, QImage::Format_RGB32);

    ElementInfo<ImageType> elementInfo1(image1);
    ElementInfo<ImageType> elementInfo2(image2);

    // 插入第一个元素
    manager.insertElement(key, elementInfo1);

    // 覆盖元素
    manager.setElement(key, elementInfo2);

    // 检索元素
    auto retrievedElement = manager.getElement(key);
    ASSERT_TRUE(retrievedElement.has_value());
    EXPECT_EQ(retrievedElement->element.size(), image2.size());
}

TEST(HistoricalElementManagerTest, RetrieveNonExistentElement) {
    using KeyType = int;
    using ImageType = QImage;
    HistoricalElementManager<KeyType, ImageType> manager(10);

    KeyType key = 999;

    // 检索不存在的元素
    auto retrievedElement = manager.getElement(key);
    EXPECT_FALSE(retrievedElement.has_value());
}