#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "neuralc/tensor/storage.hpp"

namespace neuralc {
class Tensor {
  public:
    explicit Tensor(const std::vector<std::size_t>& shape);

    [[nodiscard]] const std::vector<std::size_t>& shape() const;
    [[nodiscard]] std::size_t size() const;

    // data access
    [[nodiscard]] float* data();
    [[nodiscard]] const float* data() const;

    //indexing
    [[nodiscard]] float* at(const std::vector<std::size_t>& index);
    [[nodiscard]] const float* at(const std::vector<std::size_t>& index) const;

    // creation
    [[nodiscard]] static Tensor ones(const std::vector<std::size_t>& shape);
    [[nodiscard]] static Tensor zeros(const std::vector<std::size_t>& shape);
    [[nodiscard]] Tensor clone() const;
    void fill(float value);
    void ones();
    void zeros();

    // Operators
    [[nodiscard]] Tensor operator+(const Tensor& other) const;
    [[nodiscard]] Tensor operator-(const Tensor& other) const;
    [[nodiscard]] Tensor operator*(const Tensor& other) const;
    [[nodiscard]] Tensor operator/(const Tensor& other) const;

    // Scalar operators
    [[nodiscard]] Tensor operator+(float scalar) const;
    [[nodiscard]] Tensor operator-(float scalar) const;
    [[nodiscard]] Tensor operator*(float scalar) const;
    [[nodiscard]] Tensor operator/(float scalar) const;

    //inplace operators
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    //utils
    [[nodiscard]] float sum() const;
    [[nodiscard]] float mean() const;

    //shape
    void reshape(const std::vector<std::size_t>& new_shape);
    [[nodiscard]] Tensor transpose() const;

  private:
    std::shared_ptr<Storage> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

    [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t>& index) const;

    //helpers
    void check_same_shape(const Tensor& other) const;
    void check_for_zeros(const Tensor& other) const;
    void check_same_size(const std::vector<std::size_t>& new_shape) const;
    bool same_shape(const Tensor& other) const;



};
} // namespace neuralc