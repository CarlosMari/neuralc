#include "neuralc/tensor/storage.hpp"

namespace neuralc {
Storage::Storage(std::size_t size) : data_(std::make_unique<float[]>(size)), size_(size) {}

std::size_t Storage::size() const { return size_; }

float* Storage::data() { return data_.get(); }

const float* Storage::data() const { return data_.get(); }
} // namespace neuralc