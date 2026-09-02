#include "neuralc/tensor/tensor.hpp"

#include <stdexcept>

namespace neuralc{
    Tensor::Tensor(const std::vector<std::size_t>& shape): shape_(shape){
        std::size_t size = 1;

        for (const std::size_t dimension : shape_){
            if (dimension == 0){
                throw std::invalid_argument("Tensor dimensions must be greater than zero");
            }

            size *= dimension;
        }
        storage_ = std::make_shared<Storage>(size);
    };

[[nodiscard]] const std::vector<std::size_t>& Tensor::shape() const{
    return shape_;
}

[[nodiscard]] std::size_t Tensor::size() const{
    return storage_->size();
}

[[nodiscard]] float* Tensor::data(){
    return storage_->data();
}

[[nodiscard]] const float* Tensor::data() const{
    return storage_->data();
}

[[nodiscard]] Tensor Tensor::clone() const{
    Tensor newTensor(shape_);
    newTensor.storage_ = std::make_shared<Storage>(storage_->clone());   
    return newTensor;
}

} //namespace neuralc
