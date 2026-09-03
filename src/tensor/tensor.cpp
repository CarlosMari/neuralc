#include "neuralc/tensor/tensor.hpp"

#include <stdexcept>

namespace neuralc {
Tensor::Tensor(const std::vector<std::size_t>& shape) : shape_(shape), strides_(shape.size()) {

    std::size_t size = 1;

    for (std::size_t i = shape_.size(); i-- > 0;) {
        if (shape_[i] == 0) {
            throw std::invalid_argument("Tensor dimensions must be greater than zero");
        }

        strides_[i] = size;
        size *= shape_[i];
    }

    storage_ = std::make_shared<Storage>(size);
}

[[nodiscard]] const std::vector<std::size_t>& Tensor::shape() const { return shape_; }

[[nodiscard]] std::size_t Tensor::size() const { return storage_->size(); }

[[nodiscard]] float* Tensor::data() { return storage_->data(); }

[[nodiscard]] const float* Tensor::data() const { return storage_->data(); }

[[nodiscard]] Tensor Tensor::clone() const {
    Tensor newTensor(shape_);
    newTensor.storage_ = std::make_shared<Storage>(storage_->clone());
    return newTensor;
}

[[nodiscard]] std::size_t Tensor::flat_index(const std::vector<std::size_t>& index) const {

    if (index.size() != shape_.size()) {
        throw std::out_of_range("Tensor index dimensions do not match tensor shape");
    }

    std::size_t offset = 0;

    for (std::size_t i = 0; i < index.size(); ++i) {
        if (index[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }

        offset += index[i] * strides_[i];
    }

    return offset;
}

[[nodiscard]] float* Tensor::at(const std::vector<std::size_t>& index) {

    return data() + flat_index(index);
}

[[nodiscard]] const float* Tensor::at(const std::vector<std::size_t>& index) const {

    return data() + flat_index(index);
}

void Tensor::fill(float value){
    for(std::size_t i = 0; i < size(); ++i){
        data()[i] = value;
    }
}

void Tensor::ones(){
    fill(1.0f);
}

void Tensor::zeros(){
    fill(0.0f);
}

[[nodiscard]] Tensor Tensor::ones(const std::vector<std::size_t>& shape){
       Tensor newTensor(shape);
       newTensor.fill(1.0f);
       return newTensor;
}

[[nodiscard]] Tensor Tensor::zeros(const std::vector<std::size_t>& shape){
        Tensor newTensor(shape);
        newTensor.fill(0.0f);
        return newTensor;
}

} // namespace neuralc
