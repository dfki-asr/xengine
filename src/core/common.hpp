#ifndef XENGINE_COMMON_HPP
#define XENGINE_COMMON_HPP

#include "../xengine_utils.hpp"

using namespace std;
using namespace dnnl;

float rate_MB_per_sec(const int size_in_bytes, const float time_in_ms) {
  double time_in_s = time_in_ms * 1e-3;
  double size_in_MB = size_in_bytes * 1e-6;
  return static_cast<float>(size_in_MB / time_in_s);
}

memory::format_tag get_ncdhw_tag(memory::dims &dims) {
  if (dims.size() == 5) {
    return memory::format_tag::ncdhw;
  } else if (dims.size() == 4) {
    return memory::format_tag::nchw;
  } else if (dims.size() == 3) {
    return memory::format_tag::ncw;
  } else if (dims.size() == 2) {
    return memory::format_tag::nc;
  } else {
    throw runtime_error("number of dims not supported!");
  }
}

float maybe_do_reorder(memory &src, memory &dst, stream &dst_stream,
                       const int measure_time) {
  auto sameEngine = (src.get_engine() == dst.get_engine());
  auto sameFormat = (src.get_desc() == dst.get_desc());
  stream s = dst_stream;
  if (sameEngine) {
    if (sameFormat) {
      dst = src;
      return 0;
    }
  } else {
    if (dst_stream.get_engine().get_kind() == engine::kind::cpu) {
      s = stream(src.get_engine());
    }
  }
  auto begin = get_time();
  float time_in_ms = 0;
  if (src.get_desc().dims().size() != dst.get_desc().dims().size()) {
    auto src_dims = src.get_desc().dims();
    auto dst_dims = dst.get_desc().dims();
    if (src_dims.size() > dst_dims.size()) {
      memory::dims flat_dims = {src_dims.begin(),
                                src_dims.begin() + dst_dims.size()};
      auto flat_desc =
          memory::desc(flat_dims, g_data_type, memory::format_tag::nc);
      memory flat_src = make_memory(flat_desc, src.get_engine());
      // copy data from src to flat_src
      read_from_dnnl_memory(src.get_data_handle(), flat_src);
      reorder(flat_src, dst)
          .execute(s, {{DNNL_ARG_FROM, flat_src}, {DNNL_ARG_TO, dst}});
      s.wait();
    } else {
      throw runtime_error("Cannot reorder lower dims to higher dims!");
    }
  } else {
    reorder(src, dst).execute(s, {{DNNL_ARG_FROM, src}, {DNNL_ARG_TO, dst}});
    s.wait();
  }
  if (measure_time) {
    time_in_ms = get_elapsed_ms(begin);
  }
  return time_in_ms;
}

void add_reorder(memory &src, memory &dst, vector<unique_ptr<primitive>> &p,
                 vector<unordered_map<int, memory>> &p_args) {
  p.push_back(move(make_unique<primitive>(reorder(src, dst))));
  p_args.push_back(
      unordered_map<int, memory>({{DNNL_ARG_FROM, src}, {DNNL_ARG_TO, dst}}));
}

void execute_primitives(vector<unique_ptr<primitive>> &p,
                        vector<unordered_map<int, memory>> &p_args,
                        const engine &eng) {
  stream s = stream(eng);
  for (size_t i = 0; i < p.size(); i++) {
    p.at(i)->execute(s, p_args.at(i));
  }
  s.wait();
}
#endif