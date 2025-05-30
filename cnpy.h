// cnpy.h
// 用于读取numpy的npy格式文件的简单C++库

#ifndef CNPY_H_
#define CNPY_H_

#include <stdint.h>
#include <zlib.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cnpy {

struct NpyArray {
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size,
             bool _fortran_order)
        : shape(_shape),
          word_size(_word_size),
          fortran_order(_fortran_order),
          num_vals(1) {
        for (size_t i = 0; i < shape.size(); i++) num_vals *= shape[i];
        data_holder = std::shared_ptr<std::vector<char>>(
            new std::vector<char>(num_vals * word_size));
    }

    NpyArray() : shape(0), word_size(0), fortran_order(false), num_vals(0) {}

    template <typename T>
    T* data() {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template <typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(&(*data_holder)[0]);
    }

    template <typename T>
    std::vector<T> as_vec() const {
        const T* p = data<T>();
        return std::vector<T>(p, p + num_vals);
    }

    size_t num_bytes() const { return word_size * num_vals; }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    size_t num_vals;
};

void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape,
                      bool& fortran_order);
void parse_npy_header(unsigned char* buffer, size_t& word_size,
                      std::vector<size_t>& shape, bool& fortran_order);
NpyArray npy_load(std::string fname);
}  // namespace cnpy

#endif  // CNPY_H_