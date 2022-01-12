#ifndef XENGINE_UTILS_HPP
#define XENGINE_UTILS_HPP

#include "dnnl.hpp"
#include <chrono>
#include <string>

using namespace std;

dnnl::memory::data_type g_data_type = dnnl::memory::data_type::f32;

inline dnnl::memory make_memory(const dnnl::memory::desc &md,
                                const dnnl::engine &eng) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    return dnnl::memory(md, eng);
  } else {
    return dnnl::sycl_interop::make_memory(
        md, eng, dnnl::sycl_interop::memory_kind::buffer);
  }
#else
  return dnnl::memory(md, eng);
#endif
}

inline dnnl::memory make_memory(const dnnl::memory::desc &md,
                                const dnnl::engine &eng, void *handle) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    return dnnl::memory(md, eng, handle);
  } else {
    return dnnl::sycl_interop::make_memory(
        md, eng, dnnl::sycl_interop::memory_kind::buffer, handle);
  }
#else
  return dnnl::memory(md, eng, handle);
#endif
}

inline uint8_t _rand_uint8_t() {
  // Generates a random int number between 0 and 100
  return static_cast<uint8_t>(rand() % 100 + 1);
}

inline std::vector<dnnl::memory::dim> get_output_dims(
    const dnnl::memory::dims &dims, const dnnl::memory::dim &channels,
    const dnnl::memory::dims &kernel, const dnnl::memory::dims &stride,
    const dnnl::memory::dims &padding_l, const dnnl::memory::dims &padding_r) {
  std::vector<dnnl::memory::dim> output_dims;
  output_dims.reserve(dims.size());
  auto batch_size = dims.at(0);
  auto offset = 2;
  output_dims.push_back(batch_size);
  output_dims.push_back(channels);
  for (size_t i = 0; i < kernel.size(); ++i) {
    auto const input_value = dims.at(i + offset);
    auto const value = static_cast<dnnl::memory::dim>(
        ((input_value - kernel.at(i) + padding_l.at(i) + padding_r.at(i)) /
         static_cast<float>(stride.at(i))) +
        1);
    output_dims.push_back(value);
  }
  return static_cast<dnnl::memory::dims>(output_dims);
}

std::chrono::high_resolution_clock::time_point get_time() {
  return std::chrono::high_resolution_clock::now();
}

float get_elapsed_ms(std::chrono::high_resolution_clock::time_point begin) {
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  return static_cast<float>(duration) * 1e-6;
}

void print_memory_usage(const string memory_file = "",
                        const string event_info = "") {
  string cmd = "(free mem --mega | grep -v total | tr -s ' ' | cut -d ' ' -f 3 "
               "| head -n 1 | tr -s \"'\n'\" ' '; echo " +
               event_info + ")";
  if (!memory_file.empty())
    cmd += " >> " + memory_file;
  system(cmd.c_str());
}

std::vector<std::vector<std::string>>
parse_input_file(const std::string &path) {
  auto info = std::vector<std::vector<std::string>>();
  std::ifstream ifs(path, std::ifstream::in);
  if (ifs.is_open()) {
    std::string line;
    while (getline(ifs, line)) {
      std::vector<std::string> results;
      std::string token;
      std::istringstream tokenStream(line.c_str());
      while (getline(tokenStream, token, ';')) {
        results.push_back(token);
      }
      info.push_back(results);
    }
    ifs.close();
  } else {
    throw std::runtime_error("Could not open file!");
  }
  return info;
}

int checkIfFileExists(const std::string &path) {
  std::ifstream ifs(path, std::ifstream::in);
  if (ifs.is_open()) {
    ifs.close();
    return 1;
  }
  return 0;
}

void writeString2File(const std::string &file_path, const std::string &s) {
  std::ofstream f;
  f.open(file_path);
  f << s;
  f.close();
}

#endif