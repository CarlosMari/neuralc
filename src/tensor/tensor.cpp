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


[[nodiscard]] bool Tensor::same_shape(const Tensor& other) const {
    return other.shape_ == shape_;
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

// helpers
void Tensor::check_same_shape(const Tensor& other) const {
    if (other.shape_ != shape_) {
        throw std::invalid_argument("Tensors must have the same shape");
    }
}

void Tensor::check_same_size(
    const std::vector<std::size_t>& new_shape) const {

    std::size_t new_size = 1;

    for (std::size_t dimension : new_shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be greater than zero");
        }

        new_size *= dimension;
    }

    if (new_size != size()) {
        throw std::invalid_argument(
            "New shape must contain the same number of elements");
    }
}

void Tensor::check_for_zeros(const Tensor& other) const {
   for (std::size_t i = 0; i < other.size(); ++i) {
        if (other.data()[i] == 0.0f){
            throw std::invalid_argument("Cannot divide by zero");
        }
    } 
}

//factories
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

// Operators
[[nodiscard]] Tensor Tensor::operator+(const Tensor& other) const {
    
    check_same_shape(other);    
    Tensor newTensor(shape_);

    for (std::size_t i = 0; i < size(); ++i) {
        newTensor.data()[i] = data()[i] + other.data()[i];
    }

    return newTensor;
}

[[nodiscard]] Tensor Tensor::operator-(const Tensor& other) const {

    check_same_shape(other);
    Tensor newTensor(shape_);

    for (std::size_t i = 0; i < size(); ++i) {
        newTensor.data()[i] = data()[i] - other.data()[i];
    }

    return newTensor;
}

[[nodiscard]] Tensor Tensor::operator*(const Tensor& other) const {
    
    check_same_shape(other);
    Tensor newTensor(shape_);

    for (std::size_t i = 0; i < size(); ++i) {
        newTensor.data()[i] = data()[i] * other.data()[i];
    }

    return newTensor;
}

[[nodiscard]] Tensor Tensor::operator/(const Tensor& other) const {
    
    check_same_shape(other);
    check_for_zeros(other);
    Tensor newTensor(shape_);

    for (std::size_t i = 0; i < size(); ++i) {
        newTensor.data()[i] = data()[i] / other.data()[i];
    }

    return newTensor;
}

// Scalar Operators

[[nodiscard]] Tensor Tensor::operator*(float scalar) const{
    Tensor newTensor(shape_);
    newTensor.fill(scalar);
    return *this * newTensor;
}

[[nodiscard]] Tensor Tensor::operator+(float scalar) const{
    Tensor newTensor(shape_);
    newTensor.fill(scalar);
    return *this + newTensor;
}

[[nodiscard]] Tensor Tensor::operator-(float scalar) const{
    Tensor newTensor(shape_);
    newTensor.fill(scalar);
    return *this - newTensor;
}

[[nodiscard]] Tensor Tensor::operator/(float scalar) const{
    Tensor newTensor(shape_);
    newTensor.fill(scalar);
    return *this / newTensor;
}


//inplace operators
Tensor& Tensor::operator+=(const Tensor& other){
    check_same_shape(other);
    for (std::size_t i = 0; i < size(); ++i) {
        data()[i] += other.data()[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){
    check_same_shape(other);
    for (std::size_t i = 0; i < size(); ++i) {
        data()[i] -= other.data()[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other){
    check_same_shape(other);
    for (std::size_t i = 0; i < size(); ++i) {
        data()[i] *= other.data()[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other){
    check_same_shape(other);
    check_for_zeros(other);
    for (std::size_t i = 0; i < size(); ++i) {
        data()[i] /= other.data()[i];
    }
    return *this;
}


[[nodiscard]] float Tensor::sum() const{
    float total = 0.0f;
    for (std::size_t i = 0; i < size(); ++i) {
        total += data()[i];
    }

    return total;
}

[[nodiscard]] float Tensor::mean() const{
    return sum() / static_cast<float>(size());
}

void Tensor::reshape(const std::vector<std::size_t>& new_shape) {
    check_same_size(new_shape);

    shape_ = new_shape;
    strides_.resize(new_shape.size());

    std::size_t stride = 1;

    for (std::size_t i = new_shape.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= new_shape[i];
    }
}

[[nodiscard]] Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Transpose currently only supports 2D tensors");
    }

    std::vector<std::size_t> new_shape{
        shape_[1],
        shape_[0]
    };

    Tensor newTensor(new_shape);

    for (std::size_t i = 0; i < shape_[0]; ++i) {
        for (std::size_t j = 0; j < shape_[1]; ++j) {
            std::size_t old_index = i * strides_[0] + j * strides_[1];

            std::size_t new_index = j * newTensor.strides_[0]
                                  + i * newTensor.strides_[1];

            newTensor.data()[new_index] = data()[old_index];
        }
    }

    return newTensor;
}

} // namespace neuralc
