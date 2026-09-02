#include "neuralc/tensor/storage.hpp"

#include <algorithm>

namespace neuralc {
Storage::Storage(std::size_t size) : data_(std::make_unique<float[]>(size)), size_(size) {}

std::size_t Storage::size() const { return size_; }

float* Storage::data() { return data_.get(); }

const float* Storage::data() const { return data_.get(); }

[[nodiscard]] Storage Storage::clone() const {
    Storage new_storage(size_);

    std::copy_n(data(), size_, new_storage.data());

    return new_storage;
}

} // namespace neuralc