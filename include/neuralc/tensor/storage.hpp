#pragma once

#include <cstddef>
#include <memory>

namespace neuralc{

    class Storage{
        public:
            explicit Storage(std::size_t size);

            [[nodiscard]] std::size_t size() const;

            [[nodiscard]] float* data();
            [[nodiscard]] const float* data() const;

        private:
            std::unique_ptr<float[]> data_;
            std::size_t size_;
    };
} //namespace neuralc
