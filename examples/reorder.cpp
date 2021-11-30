#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../src/example_utils.hpp"
#include "dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void reorder_example(dnnl::engine &src_eng, dnnl::engine &dst_eng,
                     dnnl::stream &s) {
  const memory::dim N = 3, IC = 3, IH = 227, IW = 227;
  memory::dims src_dims = {N, IC, IH, IW};

  // Allocate buffers.
  std::vector<float> src_data(product(src_dims));
  std::vector<float> dst_data(product(src_dims));

  // Initialize src tensor.
  std::generate(src_data.begin(), src_data.end(), []() {
    static int i = 0;
    return std::cos(i++ / 10.f);
  });

  // Create memory descriptors and memory objects for src and dst.
  auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
  auto dst_md = memory::desc(src_dims, dt::f32, tag::nhwc);
  auto src_mem = memory(src_md, src_eng);
  auto dst_mem = memory(dst_md, dst_eng);

  // Write data to memory object's handle.
  write_to_dnnl_memory(src_data.data(), src_mem);

  // Create primitive descriptor.
  auto reorder_pd = reorder::primitive_desc(src_eng, src_md, dst_eng, dst_md);
  auto reorder_prim = reorder(reorder_pd);
  std::unordered_map<int, memory> reorder_args;
  reorder_args.insert({DNNL_ARG_SRC, src_mem});
  reorder_args.insert({DNNL_ARG_DST, dst_mem});

  // Execute reorder
  reorder_prim.execute(s, reorder_args);
  s.wait();

  read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "usage: ./examples/reorder NUM_GPUS" << std::endl;
    return 1;
  }
  dnnl::engine cpu(dnnl::engine::kind::cpu, 0);
  dnnl::stream cpu_stream(cpu);
  size_t num_gpus = std::stoi(argv[1]);
  for (size_t i = 0; i < num_gpus; i++) {
    dnnl::engine gpu0(dnnl::engine::kind::gpu, i);
    dnnl::stream gpu0_stream(gpu0);
    reorder_example(cpu, gpu0, gpu0_stream);
    reorder_example(gpu0, cpu, gpu0_stream);
  }
}