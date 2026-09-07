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

TEST(TensorTest, Addition) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(2.0f);
    b.fill(3.0f);

    Tensor c = a + b;

    EXPECT_EQ(c.shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_EQ(c.size(), 6);

    for (std::size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], 5.0f);
    }
}

TEST(TensorTest, Subtraction) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(5.0f);
    b.fill(2.0f);

    Tensor c = a - b;

    EXPECT_EQ(c.shape(), std::vector<std::size_t>({2, 3}));

    for (std::size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], 3.0f);
    }
}

TEST(TensorTest, Multiplication) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(2.0f);
    b.fill(3.0f);

    Tensor c = a * b;

    EXPECT_EQ(c.shape(), std::vector<std::size_t>({2, 3}));

    for (std::size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], 6.0f);
    }
}

TEST(TensorTest, Division) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(6.0f);
    b.fill(2.0f);

    Tensor c = a / b;

    EXPECT_EQ(c.shape(), std::vector<std::size_t>({2, 3}));

    for (std::size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c.data()[i], 3.0f);
    }
}

TEST(TensorTest, DivisionByZeroThrows) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(6.0f);
    b.fill(2.0f);

    b.data()[3] = 0.0f;

    EXPECT_THROW(static_cast<void>(a / b), std::invalid_argument);
}


TEST(TensorTest, ScalarAddition) {
    Tensor tensor({2, 3});
    tensor.fill(2.0f);

    Tensor result = tensor + 3.0f;

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], 5.0f);
        EXPECT_FLOAT_EQ(tensor.data()[i], 2.0f);
    }
}

TEST(TensorTest, ScalarSubtraction) {
    Tensor tensor({2, 3});
    tensor.fill(5.0f);

    Tensor result = tensor - 2.0f;

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], 3.0f);
        EXPECT_FLOAT_EQ(tensor.data()[i], 5.0f);
    }
}

TEST(TensorTest, ScalarMultiplication) {
    Tensor tensor({2, 3});
    tensor.fill(3.0f);

    Tensor result = tensor * 4.0f;

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], 12.0f);
        EXPECT_FLOAT_EQ(tensor.data()[i], 3.0f);
    }
}

TEST(TensorTest, ScalarDivision) {
    Tensor tensor({2, 3});
    tensor.fill(12.0f);

    Tensor result = tensor / 4.0f;

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], 3.0f);
        EXPECT_FLOAT_EQ(tensor.data()[i], 12.0f);
    }
}

TEST(TensorTest, ScalarDivisionByZeroThrows) {
    Tensor tensor({2, 3});
    tensor.fill(6.0f);

    EXPECT_THROW(static_cast<void>(tensor / 0.0f), std::invalid_argument);
}

TEST(TensorTest, InPlaceAddition) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(2.0f);
    b.fill(3.0f);

    Tensor& result = (a += b);

    EXPECT_EQ(&result, &a);

    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], 5.0f);
    }
}

TEST(TensorTest, InPlaceSubtraction) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(5.0f);
    b.fill(2.0f);

    Tensor& result = (a -= b);

    EXPECT_EQ(&result, &a);

    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], 3.0f);
    }
}

TEST(TensorTest, InPlaceMultiplication) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(2.0f);
    b.fill(3.0f);

    Tensor& result = (a *= b);

    EXPECT_EQ(&result, &a);

    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], 6.0f);
    }
}

TEST(TensorTest, InPlaceDivision) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(6.0f);
    b.fill(2.0f);

    Tensor& result = (a /= b);

    EXPECT_EQ(&result, &a);

    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], 3.0f);
    }
}

TEST(TensorTest, InPlaceDivisionByZeroThrows) {
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(6.0f);
    b.fill(2.0f);

    b.data()[3] = 0.0f;

    EXPECT_THROW(static_cast<void>(a /= b), std::invalid_argument);

    // The important part: a was not partially modified.
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], 6.0f);
    }
}

TEST(TensorTest, Sum) {
    Tensor tensor({2, 3});

    tensor.data()[0] = 1.0f;
    tensor.data()[1] = 2.0f;
    tensor.data()[2] = 3.0f;
    tensor.data()[3] = 4.0f;
    tensor.data()[4] = 5.0f;
    tensor.data()[5] = 6.0f;

    EXPECT_FLOAT_EQ(tensor.sum(), 21.0f);
}

TEST(TensorTest, SumWithNegativeValues) {
    Tensor tensor({2, 2});

    tensor.data()[0] = -1.0f;
    tensor.data()[1] = 2.0f;
    tensor.data()[2] = -3.0f;
    tensor.data()[3] = 4.0f;

    EXPECT_FLOAT_EQ(tensor.sum(), 2.0f);
}

TEST(TensorTest, SumSingleElement) {
    Tensor tensor({1});

    tensor.data()[0] = 42.0f;

    EXPECT_FLOAT_EQ(tensor.sum(), 42.0f);
}

TEST(TensorTest, Mean) {
    Tensor tensor({2, 3});

    tensor.data()[0] = 1.0f;
    tensor.data()[1] = 2.0f;
    tensor.data()[2] = 3.0f;
    tensor.data()[3] = 4.0f;
    tensor.data()[4] = 5.0f;
    tensor.data()[5] = 6.0f;

    EXPECT_FLOAT_EQ(tensor.mean(), 3.5f);
}

TEST(TensorTest, MeanWithNegativeValues) {
    Tensor tensor({2, 2});

    tensor.data()[0] = -1.0f;
    tensor.data()[1] = 2.0f;
    tensor.data()[2] = -3.0f;
    tensor.data()[3] = 4.0f;

    EXPECT_FLOAT_EQ(tensor.mean(), 0.5f);
}

TEST(TensorTest, MeanSingleElement) {
    Tensor tensor({1});

    tensor.data()[0] = 42.0f;

    EXPECT_FLOAT_EQ(tensor.mean(), 42.0f);
}


TEST(TensorTest, Reshape) {
    Tensor tensor({2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor.data()[i] = static_cast<float>(i + 1);
    }

    tensor.reshape({3, 2});

    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({3, 2}));

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], static_cast<float>(i + 1));
    }

    EXPECT_FLOAT_EQ(*tensor.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(*tensor.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(*tensor.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(*tensor.at({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(*tensor.at({2, 0}), 5.0f);
    EXPECT_FLOAT_EQ(*tensor.at({2, 1}), 6.0f);
}

TEST(TensorTest, ReshapeChangesDimensionality) {
    Tensor tensor({2, 3});

    tensor.reshape({6});

    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({6}));
    EXPECT_EQ(tensor.size(), 6);
}

TEST(TensorTest, ReshapeRejectsDifferentSize) {
    Tensor tensor({2, 3});

    EXPECT_THROW(tensor.reshape({2, 2}), std::invalid_argument);
}

TEST(TensorTest, ReshapeRejectsZeroDimension) {
    Tensor tensor({2, 3});

    EXPECT_THROW(tensor.reshape({0, 6}), std::invalid_argument);
}

TEST(TensorTest, FailedReshapeDoesNotModifyTensor) {
    Tensor tensor({2, 3});

    tensor.fill(5.0f);

    EXPECT_THROW(tensor.reshape({4, 2}), std::invalid_argument);

    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({2, 3}));

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], 5.0f);
    }
}

TEST(TensorTest, Transpose) {
    Tensor tensor({2, 3});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor.data()[i] = static_cast<float>(i + 1);
    }

    Tensor result = tensor.transpose();

    EXPECT_EQ(result.shape(), std::vector<std::size_t>({3, 2}));

    EXPECT_FLOAT_EQ(*result.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(*result.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(*result.at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(*result.at({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(*result.at({2, 0}), 3.0f);
    EXPECT_FLOAT_EQ(*result.at({2, 1}), 6.0f);
}

} // namespace neuralc

