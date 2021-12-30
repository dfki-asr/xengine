#include "core/device.hpp"
#include "core/operator_list.hpp"
#include "core/tensor.hpp"
#include "onnx/checker.h"
#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"
#include <png.h>

using namespace std;
using namespace onnx;
using namespace google::protobuf;

using ft = memory::format_tag;
using dt = memory::data_type;

ModelProto loadModel(const string &path) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  ifstream f(path, ios_base::binary);
  ModelProto model;
  model.ParseFromIstream(&f);
  f.close();
  checker::check_model(model);
  shape_inference::InferShapes(model);
  checker::check_model(model);
  return model;
}

enum class Format {
  Uint8 = 0x08,
  Int8 = 0x09,
  Int16 = 0x0B,
  Int32 = 0x0C,
  Float = 0x0D,
  Double = 0x0E
};

template <typename T>
T read_number(ifstream &in, size_t &offset, const uint8_t &littleEndian) {
  T number;
  in.seekg(offset);
  in.read(reinterpret_cast<char *>(&number), sizeof(number));
  auto it = reinterpret_cast<char *>(&number);
  if (!littleEndian) {
    reverse(it, it + sizeof(number));
  }
  offset += sizeof(number);
  return number;
}

void read_header(ifstream &in, vector<uint32_t> &shape, uint8_t &littleEndian,
                 uint8_t &format) {
  size_t offset = 0;
  auto const magicNumber = read_number<uint16_t>(in, offset, 0);
  assert(magicNumber == 0 || magicNumber == 1);
  littleEndian = (magicNumber != 0);
  format = read_number<uint8_t>(in, offset, littleEndian);
  auto const numDims = read_number<uint8_t>(in, offset, littleEndian);
  for (int i = 0; i < numDims; ++i) {
    shape.push_back(read_number<uint32_t>(in, offset, littleEndian));
  }
  auto file_offset = sizeof(uint32_t) + numDims * sizeof(uint32_t);
  in.seekg(file_offset);
}

void normalizeImages(size_t const &batchSize, size_t const &channels,
                     size_t const &size, float *buffer) {
  vector<double> NORMALIZATION_MEAN{0.485, 0.456, 0.406};
  vector<double> NORMALIZATION_SIGMA{0.229, 0.224, 0.225};
  vector<double> sigma_factor{1.0f / NORMALIZATION_SIGMA[0],
                              1.0f / NORMALIZATION_SIGMA[1],
                              1.0f / NORMALIZATION_SIGMA[2]};
  for (size_t b = 0; b < batchSize; b++) {
    for (size_t c = 0; c < channels; c++) {
      auto minmax = minmax_element(buffer, buffer + size);
      const double min_value = *(minmax.first);
      const double max_value = *(minmax.second);
      for (size_t j = 0; j < size; j++) {
        float x = (buffer[j] - min_value) /
                  (max_value - min_value); // read x and scale to range  0 ... 1
        auto norm = static_cast<float>(sigma_factor[c] *
                                       (x - float(NORMALIZATION_MEAN[c])));
        buffer[j] = static_cast<float>(norm);
      }
      buffer += size;
    }
  }
}

void writeImageAsPNG(vector<uint8_t> &imageBuffer, vector<uint32_t> &shape,
                     const string &imgpath, size_t sampleIdx) {
  int i = shape.size() - 1;
  int inChannel = shape.size() > 3 ? shape.at(1) : 1;
  int inHeight = shape.at(i - 1);
  int inWidth = shape.at(i);
  int imgSize = inChannel * inHeight * inWidth;
  png_structp png = nullptr;
  png_infop info = nullptr;
  FILE *fp;
  try {
    fp = fopen(imgpath.c_str(), "wb");
    if (fp == nullptr) {
      throw runtime_error("Could not open file " + imgpath + "!");
    }
    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr,
                                  nullptr);
    if (!png) {
      throw runtime_error("Could not create png read struct.");
    }
    info = png_create_info_struct(png);
    if (!info) {
      throw runtime_error("Could not create info read struct.");
    }
    setjmp(png_jmpbuf(png));
    png_init_io(png, fp);
    png_byte color_type;
    if (inChannel == 1) {
      color_type = PNG_COLOR_TYPE_GRAY;
    } else if (inChannel == 3) {
      color_type = PNG_COLOR_TYPE_RGB;
    } else {
      throw runtime_error(
          "Unsupported channel number: " + to_string(inChannel) + " channels");
    }
    int bits = sizeof(uint8_t) * 8;
    png_set_IHDR(png, info, inWidth, inHeight, bits, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    int sample_offset = sampleIdx * imgSize;
    int layout_NHWC = 0;
    if (layout_NHWC) {
      for (int y = 0; y < inHeight; y++) {
        png_write_row(
            png, reinterpret_cast<png_bytep>(
                     &imageBuffer[sample_offset + y * inWidth * inChannel]));
      }
    } else {
      png_bytep *row_ptr = new png_bytep[inHeight];
      for (int y = 0; y < inHeight; y++) {
        row_ptr[y] = reinterpret_cast<png_bytep>(
            new unsigned char *[inWidth * inChannel]);
        for (int c = 0; c < inChannel; c++) {
          for (int x = 0; x < inWidth; x++) {
            row_ptr[y][x * inChannel + c] = reinterpret_cast<png_byte>(
                imageBuffer
                    .data()[sample_offset + (c * inHeight + y) * inWidth + x]);
          }
        }
      }
      png_write_image(png, row_ptr);
    }
    png_write_end(png, nullptr);
  } catch (const string &msg) {
    throw runtime_error("Error reading file " + imgpath);
  }
  png_destroy_write_struct(&png, &info);
  if (fp != nullptr) {
    fclose(fp);
  }
}

template <typename T>
void load_data_to_handle(float *buffer, const size_t &batch,
                         const size_t &batchsize, const string &file_path,
                         int writeImages = 0) {
  ifstream in(file_path, ifstream::binary);
  if (!in.is_open())
    throw runtime_error("Could not open file " + file_path);

  vector<uint32_t> shape;
  uint8_t littleEndian, format;
  read_header(in, shape, littleEndian, format);
  Format f = static_cast<Format>(format);
  assert(f == Format::Uint8 || f == Format::Int16);
  size_t numElements = batchsize;
  for (size_t i = 1; i < shape.size(); ++i) {
    numElements *= shape.at(i);
  }
  const auto sizeInBytes = numElements * sizeof(T);
  const auto totalElements = shape.at(0);
  if (batchsize > totalElements) {
    throw runtime_error("ERROR: Requested batchsize " + to_string(batchsize) +
                        " is greater than number of elements in file (" +
                        to_string(totalElements) + ")!");
  }
  auto tmpBuffer = unique_ptr<T[]>(new T[numElements]);
  long unsigned int pos = in.tellg();
  in.seekg(pos + (batch * sizeInBytes));
  in.read(reinterpret_cast<char *>(tmpBuffer.get()), sizeInBytes);
  transform(tmpBuffer.get(), tmpBuffer.get() + numElements, buffer,
            [](T a) { return static_cast<float>(a); });

  if (!writeImages)
    return;
  for (size_t sampleIdx = 0; sampleIdx < batchsize; sampleIdx++) {
    if (shape.size() > 2) {
      auto imageBuffer = vector<uint8_t>(numElements);
      transform(tmpBuffer.get(), tmpBuffer.get() + numElements,
                imageBuffer.data(), [](T a) { return a; });
      const string out_dir = "output/";
      const string imgpath =
          out_dir + to_string(batch) + "_" + to_string(sampleIdx) + ".png";
      writeImageAsPNG(imageBuffer, shape, imgpath, sampleIdx);
    } else {
      cout << buffer[sampleIdx] << endl;
    }
  }
}

inline memory::dims _get_dim_from_shape(const TensorShapeProto &shape) {
  auto dims = vector<memory::dim>();
  for (auto i : shape.dim()) {
    dims.push_back(static_cast<memory::dim>(i.dim_value()));
  }
  return static_cast<memory::dims>(dims);
}

inline memory::dims
_get_dim_from_shape(const RepeatedField<long int> &proto_dims) {
  auto dims = vector<memory::dim>();
  for (auto i : proto_dims) {
    dims.push_back(static_cast<memory::dim>(i));
  }
  return static_cast<memory::dims>(dims);
}

inline ft _get_tag(const int &firstNode, const int &num_dims,
                   const int &weights) {
  if (firstNode && weights) {
    // input
    if (num_dims == 4) {
      return ft::nchw;
    } else if (num_dims == 2) {
      return ft::nc;
    } else if (num_dims == 5) {
      return ft::ncdhw;
    } else {
      throw runtime_error("unsupported tensor!");
    }
  }
  if (num_dims == 4) {
    if (weights) {
      return ft::oihw;
    } else {
      return ft::nchw;
    }
  } else if (num_dims == 5) {
    if (weights) {
      return ft::oidhw;
    } else {
      return ft::ncdhw;
    }
  } else if (num_dims == 2) {
    if (weights) {
      return ft::oi;
    } else {
      return ft::nc;
    }
  } else if (num_dims == 1) {
    return ft::x;
  } else {
    throw runtime_error("unsupported tensor!");
  }
}

inline float _rand_float() {
  // Generates a random float number between 0 and 1
  return static_cast<float>(rand() / static_cast<float>(RAND_MAX));
}

inline void
insert_nonInput_tensors(unordered_map<string, shared_ptr<Tensor>> &tensors,
                        const RepeatedPtrField<ValueInfoProto> &onnx_info) {
  for (auto info : onnx_info) {
    const auto name = info.name();
    const auto shape = info.type().tensor_type().shape();
    const auto dims = _get_dim_from_shape(shape);
    tensors.emplace(name, move(make_shared<Tensor>(name, dims)));
  }
}

inline void
insert_input_tensors(unordered_map<string, shared_ptr<Tensor>> &tensors,
                     const GraphProto &graph) {
  auto onnx_info = graph.input();
  // Input
  auto first_node = onnx_info[0];
  const auto first_node_name = first_node.name();
  const auto first_node_dims =
      _get_dim_from_shape(first_node.type().tensor_type().shape());
  tensors.emplace(first_node_name,
                  move(make_shared<Tensor>(first_node_name, first_node_dims)));
  // Labels
  auto last_node = graph.output()[0];
  const auto last_node_dims =
      _get_dim_from_shape(last_node.type().tensor_type().shape());
  memory::dims label_dims = memory::dims({last_node_dims.at(0), 1});
  tensors.emplace("labels", move(make_shared<Tensor>("labels", label_dims)));
  // Parameters
  auto initializers = graph.initializer();
  assert(initializers.size() == onnx_info.size() - 1);
  for (auto i = 0; i < initializers.size(); i++) {
    auto onnx_info = initializers[i];
    assert(onnx_info.data_type() == 1);
    const auto name = onnx_info.name();
    const memory::dims dims = _get_dim_from_shape(onnx_info.dims());
    tensors.emplace(name, move(make_shared<Tensor>(name, dims)));
  }
}

void fillTensors(unordered_map<string, shared_ptr<Tensor>> &tensors,
                 const ModelProto &model) {
  insert_nonInput_tensors(tensors, model.graph().value_info());
  insert_nonInput_tensors(tensors, model.graph().output());
  insert_input_tensors(tensors, model.graph());
  tensors.emplace("loss",
                  move(make_shared<Tensor>("loss", tensors["labels"]->dims())));
}

void maxMemoryDemandInfo(vector<shared_ptr<Operator>> &operators,
                         const int training, const int verbose = 0) {
  float op_md = 0.0f;
  for (auto op : operators) {
    float bytes = op->getFwdMemoryConsumption();
    if (verbose > 1) {
      cout << op->name << " fwd with " << to_string(bytes / (1024.0f * 1024.0f))
           << " MB." << endl;
    }
    op_md += bytes;
  }
  if (training) {
    for (size_t i = 0; i < operators.size(); i++) {
      auto opID = operators.size() - i - 1;
      float bytes = operators.at(opID)->getBwdMemoryConsumption();
      if (verbose > 1) {
        cout << operators.at(opID)->name << " bwd with "
             << to_string(bytes / (1024.0f * 1024.0f)) << " MB." << endl;
      }
      op_md += bytes;
    }
  }
  op_md /= (1024.0f * 1024.0f);
  cout << "max memory if keeping all operators at the same time: "
       << to_string(op_md) << " MB." << endl;
}

void maxMemoryDemandInfo(unordered_map<string, shared_ptr<Tensor>> &tensors,
                         const int verbose = 0) {
  float tensor_md = 0.0f;
  for (auto it = tensors.begin(); it != tensors.end(); it++) {
    string t_name = it->first;
    float bytes = product(it->second->dims()) * sizeof(float);
    if (verbose > 1) {
      cout << t_name << " with " << to_string(bytes / (1024.0f * 1024.0f))
           << " MB." << endl;
    }
    tensor_md += bytes;
  }
  tensor_md /= (1024.0f * 1024.0f);
  cout << "max memory if keeping all tensors at the same time: "
       << to_string(tensor_md) << " MB." << endl;
}

inline vector<string>
get_string_vector_from_proto(RepeatedPtrField<string> proto) {
  vector<string> string_vec;
  string_vec.reserve(proto.size());
  for (auto i : proto) {
    string_vec.push_back(static_cast<string>(i));
  }
  return string_vec;
}

inline vector<memory::dim> get_dim_vec_from_attr_proto(AttributeProto attr) {
  vector<memory::dim> dim_vec;
  dim_vec.reserve(attr.ints().size());
  for (auto d : attr.ints()) {
    dim_vec.push_back(static_cast<memory::dim>(d));
  }
  return dim_vec;
}

inline void
get_params_from_proto(const NodeProto &node,
                      unordered_map<string, vector<memory::dim>> &dim_params,
                      unordered_map<string, float> &float_params,
                      unordered_map<string, int> &int_params) {
  for (int i = 0; i < node.attribute_size(); i++) {
    auto attr = node.attribute(i);
    auto attr_name = attr.name();
    auto attr_type = attr.type();
    if (attr_type == 7) {
      dim_params[attr_name] = get_dim_vec_from_attr_proto(attr);
    } else if (attr_type == 1) {
      float_params[attr_name] = attr.f();
    } else if (attr_type == 2) {
      int_params[attr_name] = attr.i();
    } else {
      throw runtime_error("Unsupported ONNX Attribute Type!");
    }
  }
}

void createDevices(map<string, shared_ptr<Device>> &devices,
                   const string &device_file) {
  for (auto d : parse_input_file(device_file)) {
    const string device_name = d.at(0);
    devices[device_name] = move(make_shared<Device>(d));
  }
}
