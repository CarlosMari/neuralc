#include <gtest/gtest.h>

#include <neuralc/tensor/storage.hpp>

namespace neuralc {

TEST(StorageTest, AllocatesMemory) {
    Storage storage(10);
    EXPECT_EQ(storage.size(), 10);
    EXPECT_NE(storage.data(), nullptr);
}

TEST(StorageTest, CanReadAndWrite) {
    Storage storage(3);

    storage.data()[0] = 10.0f;
    storage.data()[1] = 20.0f;
    storage.data()[2] = 30.0f;

    EXPECT_FLOAT_EQ(storage.data()[0], 10.0f);
    EXPECT_FLOAT_EQ(storage.data()[1], 20.0f);
    EXPECT_FLOAT_EQ(storage.data()[2], 30.0f);
}

} // namespace neuralc