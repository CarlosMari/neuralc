#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <neuralc/tensor/tensor.hpp>

namespace neuralc {

TEST(TensorTest, CreatesTensor) {
    Tensor tensor({2, 3});
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, DataCloneTest) {
    Tensor a({2, 3});
    Tensor b = a;
    Tensor c = a.clone();

    c.data()[0] = 30.0f;
    b.data()[0] = 42.0f;

    EXPECT_FLOAT_EQ(a.data()[0], 42.0f);
    EXPECT_FLOAT_EQ(c.data()[0], 30.0f);
    EXPECT_NE(a.data(), c.data());
}

} // namespace neuralc