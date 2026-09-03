#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "neuralc/tensor/storage.hpp"

namespace neuralc {
class Tensor {
  public:
    explicit Tensor(const std::vector<std::size_t>& shape);

    [[nodiscard]] static Tensor ones(const std::vector<std::size_t>& shape);
    [[nodiscard]] static Tensor zeros(const std::vector<std::size_t>& shape);

    [[nodiscard]] const std::vector<std::size_t>& shape() const;
    [[nodiscard]] std::size_t size() const;

    [[nodiscard]] float* data();
    [[nodiscard]] const float* data() const;

    [[nodiscard]] Tensor clone() const;

    [[nodiscard]] float* at(const std::vector<std::size_t>& index);
    [[nodiscard]] const float* at(const std::vector<std::size_t>& index) const;

    void fill(float value);
    void ones();
    void zeros();

  private:
    std::shared_ptr<Storage> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

    [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t>& index) const;
};
} // namespace neuralc