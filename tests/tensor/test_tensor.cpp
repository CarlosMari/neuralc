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

TEST(TensorTest, IndexTest) {
    Tensor a({2, 3});

    *a.at({0, 0}) = 20.0f;
    *a.at({1, 1}) = 40.0f;

    EXPECT_FLOAT_EQ(a.data()[0], 20.0f);
    EXPECT_FLOAT_EQ(a.data()[4], 40.0f);

    EXPECT_THROW(static_cast<void>(a.at({2, 0})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(a.at({0, 3})), std::out_of_range);
    EXPECT_THROW(static_cast<void>(a.at({0})), std::out_of_range);
}

TEST(TensorTest, ConstIndexAccess) {
    Tensor tensor({2, 3});

    *tensor.at({1, 2}) = 42.0f;

    const Tensor& const_tensor = tensor;

    EXPECT_FLOAT_EQ(*const_tensor.at({1, 2}), 42.0f);
}

TEST(TensorTest, Fill) {
    Tensor tensor({2, 3});

    tensor.fill(5.0f);

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], 5.0f);
    }
}

TEST(TensorTest, Ones) {
    Tensor tensor({2, 3});

    tensor.ones();

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], 1.0f);
    }
}

TEST(TensorTest, OnesFactory) {
    Tensor tensor = Tensor::ones({2, 3});

    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({2, 3}));

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], 1.0f);
    }
}

TEST(TensorTest, ZerosFactory) {
    Tensor tensor = Tensor::zeros({2, 3});

    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({2, 3}));

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], 0.0f);
    }
}

} // namespace neuralc