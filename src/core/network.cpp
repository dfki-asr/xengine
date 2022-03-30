#include "network.hpp"
#ifdef HAS_CBC
#include "ilp_solver_cbc.cpp"
#endif
#ifdef HAS_GUROBI
#include "ilp_solver_grb.cpp"
#endif
#if !defined(HAS_CBC) && !defined(HAS_GUROBI)
#include "ilp_solver.cpp"
#endif
#include <future>

using namespace std;
using namespace dnnl;

Network::Network(const string name, const string model_file,
                 const string device_file, const int training,
                 const string output_dir, const int verbose)
    : _name(name), _training(training), _schedule(nullptr),
      _output_dir(output_dir), _verbose(verbose), _measure_time(0),
      _opsToKeep(1), _mode(training ? "training" : "inference") {
  _devices = map<string, shared_ptr<Device>>();
  _tensors = unordered_map<string, shared_ptr<Tensor>>();
  _operators = vector<shared_ptr<Operator>>();
  _primitives = vector<unique_ptr<primitive>>();
  _primitive_args = vector<unordered_map<int, memory>>();
  _memoryLogfile =
      _output_dir + "/" + _name + "_" + _mode + "_memory_used_bytes.txt";
  if (checkIfFileExists(_memoryLogfile)) {
    string cmd = "rm " + _memoryLogfile;
    system(cmd.c_str());
  }
  createDevices(_devices, device_file);
  if (_devices.find("cpu_0") != _devices.end()) {
    _cpu_device = _devices["cpu_0"];
  } else {
    _cpu_device = move(make_shared<Device>("cpu_0", "cpu", 0, 1));
  }
  _print_memory_usage(_memoryLogfile, "init_program");
  onnx::ModelProto model = loadModel(model_file);
  fillTensors(_tensors, model);
  auto inputs = unordered_map<string, vector<string>>();
  auto outputs = unordered_map<string, vector<string>>();
  _preprocessModel(model, inputs, outputs);
  _init(model, inputs, outputs);
  if (_verbose > 0) {
    cout << endl
         << "********** " << _name << " *** " << _mode << " ********" << endl;
    maxMemoryDemandInfo(_tensors, _verbose);
    maxMemoryDemandInfo(_operators, _tensors, _training, _verbose);
  }
  auto begin = get_time();
  _fillModelParameters(model);
  const auto data_tensor_name = model.graph().input()[0].name();
  const auto labels_name = "labels";
  _tensors[data_tensor_name]->set_producer("external");
  _tensors[labels_name]->set_producer("external");
  if (_verbose > 0) {
    cout << "init took " << (get_elapsed_ms(begin)) << " ms." << endl;
  }
  _opsToKeep = 2 * _operators.size();
}

Network::~Network() {
  _primitives.clear();
  _primitive_args.clear();
  for (auto t = _tensors.begin(); t != _tensors.end(); t++) {
    t->second->release();
    t->second.reset();
  }
  _tensors.clear();
  for (auto op = _operators.begin(); op != _operators.end(); op++) {
    op->reset();
  }
  _operators.clear();
  for (auto dev = _devices.begin(); dev != _devices.end(); dev++) {
    dev->second.reset();
  }
  _devices.clear();
  _unsetSchedule();
}

/**************************************************************/
//                   public
/**************************************************************/

void Network::createSchedule(const string &schedule_file, const string &images,
                             const string &labels) {
  if (_verbose > 0) {
    cout << "Create schedule file " << schedule_file << endl;
  }
  _benchmark(images, labels);
  _writeScheduleFileMinTime(schedule_file);
}

void Network::runSchedule(const string &schedule_file, const string &images,
                          const string &labels, const size_t num_iterations) {
  if (_verbose > 0) {
    cout << "Run schedule file " << schedule_file << endl;
  }
  _setSchedule(schedule_file);
  _run(images, labels, num_iterations);
}

void Network::solveILP(const string mpsfile, const string logfile,
                       vector<pair<string, edge>> &edges,
                       vector<string> &dev_names,
                       vector<vector<float>> &compute_costs_per_op,
                       vector<float> &memory_per_op, matrix &copy_costs,
                       vector<float> &budget, vector<float> &ram) {
  string budget_str =
      to_string(static_cast<int>(budget[0] / 1024.0 / 1024.0)) + "MB_" +
      to_string(static_cast<int>(budget[1] / 1024.0 / 1024.0)) + "MB";
  string schedulefile = _output_dir + "/" + _name + "_" + _mode + "_" +
                        budget_str + "_ilp_schedule.txt";
#ifdef HAS_GUROBI
  auto ilp = ILP_Solver_GRB(_name + "_" + _mode, mpsfile, logfile, edges,
                            dev_names, compute_costs_per_op, memory_per_op,
                            copy_costs, budget, ram, _verbose);
  ilp.solve();
  ilp.printResults();
  _R = ilp.get_R();
  _S = ilp.get_S();
  _F = ilp.get_F();
  _ilpMatrices2Schedule(schedulefile);
#else
#ifdef HAS_CBC
  auto ilp = ILP_Solver_CBC(_name + "_" + _mode, mpsfile, logfile, edges,
                            dev_names, compute_costs_per_op, memory_per_op,
                            copy_costs, budget, ram, _verbose);
  ilp.solve();
  ilp.printResults();
  _R = ilp.get_R();
  _S = ilp.get_S();
  _F = ilp.get_F();
  _ilpMatrices2Schedule(schedulefile);
#else
  cout << "No ILP-solver given. Define problem as MPS only." << endl;
  auto ilp = ILP_Solver(_name + "_" + _mode, mpsfile, logfile, edges, dev_names,
                        compute_costs_per_op, memory_per_op, copy_costs, budget,
                        ram, _verbose);
#endif
#endif
}

void Network::solveILP(const string mpsfile, const string logfile,
                       const string &data_path, const string &label_path,
                       const int benchmarkILP) {
  if (_verbose > 0) {
    cout << "ILP optimizer ..." << endl;
  }
  auto compute_costs_per_op = vector<vector<float>>();
  auto edges = vector<pair<string, edge>>();
  auto memory_per_op = vector<float>();
  auto dev_names = vector<string>();
  auto budget = vector<float>();
  auto ram = vector<float>();
  size_t numOps = _training ? 2 * _operators.size() : _operators.size();
  // device names
  for (auto dev = _devices.begin(); dev != _devices.end(); dev++) {
    dev_names.push_back(dev->first);
    budget.push_back(dev->second->budget);
    ram.push_back(1.0f);
    if (_verbose > 0) {
      float budget_GB = (dev->second->budget) / 1024.0 / 1024.0 / 1024.0;
      cout << "device " << dev->first << " with " << budget_GB << " GB budget."
           << endl;
    }
  }
  // edges
  for (auto it = _tensors.begin(); it != _tensors.end(); it++) {
    auto src = it->second->producer();
    auto consumers = it->second->consumers();
    for (auto dst : consumers) {
      if (src == "external" || dst == "external") {
        continue;
      }
      if (src == dst) {
        continue;
      }
      const size_t src_idx = getOpIndexFromName(_operators, src);
      const size_t dst_idx = getOpIndexFromName(_operators, dst);
      const string edgeName = it->second->name();
      edges.push_back(pair<string, edge>(edgeName, edge(src_idx, dst_idx)));
    }
  }
  matrix copy_costs = matrix(edges.size(), _devices.size(), _devices.size(), 0);

  if (benchmarkILP) {
    if (_verbose > 0) {
      cout << "Benchmark ILP, aquire data ..." << endl;
    }
    // compute costs
    compute_costs_per_op = _benchmark("", "");
    // memory costs
    for (size_t opID = 0; opID < _operators.size(); opID++) {
      long long memory_long =
          _operators.at(opID)->getFwdMemoryConsumption(_tensors);
      float memory_float = static_cast<float>(memory_long);
      memory_per_op.push_back(memory_float);
    }
    if (_training) {
      for (size_t i = 0; i < _operators.size(); i++) {
        auto opID = _operators.size() - i - 1;
        long long memory_long =
            _operators.at(opID)->getBwdMemoryConsumption(_tensors);
        float memory_float = static_cast<float>(memory_long);
        memory_per_op.push_back(memory_float);
      }
    }
    _resetPrimitives();
    if (_devices.size() > 1) {
      cout << "measure copy costs ..." << endl;
      _fillCopyCostsMatrix(copy_costs, edges);
      vector<size_t> edges_still_uncovered =
          getUncoveredEdges(edges, copy_costs);
      if (edges_still_uncovered.size() != 0) {
        for (auto i : edges_still_uncovered) {
          cout << "edge " << edges[i].first << " uncovered!" << endl;
          size_t src = edges[i].second.get_u();
          size_t dst = edges[i].second.get_v();
          size_t opIDsrc =
              src < _operators.size() ? src : 2 * _operators.size() - src - 1;
          size_t opIDdst =
              dst < _operators.size() ? dst : 2 * _operators.size() - dst - 1;
          cout << _operators.at(opIDsrc)->name << "-->"
               << _operators.at(opIDdst)->name << endl;
        }
        throw runtime_error("still uncovered edges!");
      }
    }
    if (_verbose > 0) {
      cout << _name << " " << _mode << " has computational costs:" << endl;
      for (size_t d = 0; d < dev_names.size(); d++) {
        cout << dev_names[d] << ": ";
        for (auto c : compute_costs_per_op[d]) {
          printf("%0.3lf ms  ", c);
        }
        cout << endl;
      }
      cout << _name << " " << _mode << " has memory costs:" << endl;
      for (auto m : memory_per_op) {
        printf("%0.3lf MB  ", m / 1e6);
      }
      cout << endl;
      cout << "Edges: " << endl;
      for (auto e : edges) {
        cout << "(" << e.second.get_u() << ", " << e.second.get_v() << ")"
             << "  ";
      }
      cout << endl;
      cout << "copy costs: " << endl;
      copy_costs.print();
    }
  } else {
    // dummy data
    compute_costs_per_op.push_back(vector<float>());
    compute_costs_per_op[0] = vector<float>(numOps, 0.0f);
  }
  if (_verbose > 0) {
    cout << "Solve ILP ..." << endl;
  }

  map<string, vector<float>> budgets = {
      {"auto", budget}, // auto (default) budget
      {"3GB_3GB", {3221225472, 3221225472}},
      {"1GB_1GB", {1073741824, 1073741824}},
      {"500MB_500MB", {536870912, 536870912}},
      {"800MB_800MB", {838860800, 838860800}},
      {"850MB_500MB", {891289600, 536870912}},
      {"850MB_850MB", {891289600, 891289600}},
      {"250MB_250MB", {268435456, 268435456}},
      {"150MB_150MB", {157286400, 157286400}},
      {"100MB_100MB", {104857600, 104857600}},
      {"70MB_40MB", {73400320, 41943040}},
      {"10MB_10MB", {10485760, 10485760}}};
  vector<string> run_order = {"auto",        "3GB_3GB",     "1GB_1GB",
                              "500MB_500MB", "250MB_250MB", "150MB_150MB",
                              "100MB_100MB"};

  for (auto budget_name : run_order) {
    const string _name =
        mpsfile.substr(0, mpsfile.length() - 4) + "_budget_" + budget_name;
    const string _mpsfile = _name + ".mps";
    const string _logfile = _name + ".log";
    cout << endl
         << endl
         << endl
         << "**********************************************"
         << " solve ILP " << _mpsfile
         << "**********************************************" << endl
         << endl
         << endl
         << endl;
    solveILP(_mpsfile, _logfile, edges, dev_names, compute_costs_per_op,
             memory_per_op, copy_costs, budgets[budget_name], ram);
  }
}

void Network::solveILP(const string mpsfile, const string logfile) {
  solveILP(mpsfile, logfile, "", "", 0);
}

/**************************************************************/
//                   private
/**************************************************************/
void Network::_preprocessModel(onnx::ModelProto &model,
                               unordered_map<string, vector<string>> &inputs,
                               unordered_map<string, vector<string>> &outputs) {
  if (_tensors.empty()) {
    throw runtime_error(
        "Cannot preprocess model before tensors are initialized.");
  }
  const auto nodes = model.graph().node();
  for (const auto &node : nodes) {
    const auto name = node.name();
    inputs[name] = get_string_vector_from_proto(node.input());
    outputs[name] = get_string_vector_from_proto(node.output());
  }
  // remove Flatten operators, remove Dropout operators in non-training mode
  // mark as to-be-replaced
  auto replace_info = unordered_map<string, int>();
  for (auto i = 0; i < nodes.size(); i++) {
    const auto type = nodes.at(i).op_type();
    replace_info[nodes.at(i).name()] =
        (type == "Flatten" || type == "Reshape" ||
         (type == "Dropout" && _training == false));
  }
  for (auto i = 1; i < nodes.size() - 1; i++) {
    const auto type = nodes.at(i).op_type();
    if (replace_info[nodes.at(i).name()] == false)
      continue;
    size_t decrease = 1;
    string prev = nodes.at(i - decrease).name();
    do {
      prev = nodes.at(i - decrease).name();
      decrease += 1;
    } while (replace_info[prev] == true);
    size_t increase = 1;
    string next = nodes.at(i + increase).name();
    do {
      next = nodes.at(i + increase).name();
      increase += 1;
    } while (replace_info[next] == true);
    auto const old_outputs = outputs[prev];
    auto const old_inputs = inputs[next];
    if (old_inputs.size() > 1) {
      auto w_name = old_inputs.at(1);
      auto w_dims = _tensors[w_name]->dims();
      auto org_prev_output = old_outputs.at(0);
      auto d_dims = _tensors[org_prev_output]->dims();
      auto new_dims = vector<memory::dim>({w_dims.at(0)});
      for (size_t j = 1; j < d_dims.size(); j++)
        new_dims.push_back(d_dims.at(j));
      _tensors[w_name]->set_dims(memory::dims(new_dims));
    }
    inputs[next].at(0) = old_outputs.at(0);
    const auto name = nodes.at(i).name();
    inputs[name] = vector<string>();
    outputs[name] = vector<string>();
  }
}

void Network::_init(onnx::ModelProto &model,
                    unordered_map<string, vector<string>> &inputs,
                    unordered_map<string, vector<string>> &outputs) {
  const string fwd_prefix = "fwd_";
  const string bwd_prefix = "bwd_";
  for (const auto &node : model.graph().node()) {
    const auto name = node.name();
    if (name.empty()) {
      throw runtime_error("operator without name encountered!");
    }
    const auto type = node.op_type();
    const auto input = inputs[name];
    const auto output = outputs[name];
    auto dim_parameters = unordered_map<string, vector<memory::dim>>();
    auto float_parameters = unordered_map<string, float>();
    auto int_parameters = unordered_map<string, int>();
    get_params_from_proto(node, dim_parameters, float_parameters,
                          int_parameters);
    if (type == "Flatten" || type == "Reshape" ||
        (type == "Dropout" && _training == false))
      continue;
    if (type.find("Pool") != string::npos && _training == true) {
      auto output_name = outputs[name].at(0);
      auto ws_name = output_name + "_ws";
      _tensors[ws_name] =
          move(make_shared<Tensor>(ws_name, _tensors[output_name]->dims()));
      _tensors[ws_name]->set_producer(fwd_prefix + name);
    }
    if (type == "Conv") {
      _operators.push_back(
          make_shared<Conv>(name, input, output, dim_parameters["strides"],
                            dim_parameters["kernel_shape"],
                            dim_parameters["pads"], _tensors, _training));
    } else if (type == "ConvTranspose") {
      _operators.push_back(move(make_shared<ConvTranspose>(
          name, input, output, dim_parameters["strides"],
          dim_parameters["kernel_shape"], dim_parameters["pads"], _tensors,
          _training)));
    } else if (type == "MaxPool") {
      _operators.push_back(move(
          make_shared<MaxPool>(name, input, output, dim_parameters["strides"],
                               dim_parameters["kernel_shape"],
                               dim_parameters["pads"], _tensors, _training)));
    } else if (type == "AveragePool") {
      _operators.push_back(move(make_shared<AveragePool>(
          name, input, output, dim_parameters["strides"],
          dim_parameters["kernel_shape"], dim_parameters["pads"], _tensors,
          _training)));
    } else if (type == "GlobalAveragePool") {
      const auto i_dims = _tensors[input.at(0)]->dims();
      memory::dims kernel = {i_dims.at(2), i_dims.at(3)};
      _operators.push_back(move(make_shared<GlobalAveragePool>(
          name, input, output, kernel, _tensors, _training)));
    } else if (type == "Gemm") {
      _operators.push_back(
          move(make_shared<Gemm>(name, input, output, _tensors, _training)));
    } else if (type == "Relu") {
      _operators.push_back(
          move(make_shared<Relu>(name, input, output, _tensors, _training)));
    } else if (type == "LeakyRelu") {
      float alpha = float_parameters["alpha"];
      _operators.push_back(move(make_shared<LeakyRelu>(
          name, input, output, alpha, _tensors, _training)));
    } else if (type == "Dropout") {
      if (output.size() > 1) {
        auto mask_name = output[1];
        _tensors[mask_name] =
            move(make_shared<Tensor>(mask_name, _tensors[output[0]]->dims()));
      }
      float probability = float_parameters["ratio"];
      _operators.push_back(move(make_shared<Dropout>(
          name, input, output, probability, _tensors, _training)));
    } else if (type == "LRN") {
      if (_training) {
        auto output_name = outputs[name].at(0);
        auto ws_name = output_name + "_ws";
        _tensors[ws_name] =
            move(make_shared<Tensor>(ws_name, _tensors[output_name]->dims()));
        _tensors[ws_name]->set_producer(fwd_prefix + name);
      }
      float alpha = float_parameters["alpha"];
      float beta = float_parameters["beta"];
      float bias = float_parameters["bias"];
      int size = int_parameters["size"];
      _operators.push_back(move(make_shared<LRN>(
          name, input, output, alpha, beta, bias, size, _tensors, _training)));
    } else if (type == "BatchNormalization") {
      if (_training) {
        auto gamma_diff_name =
            "diff_" + inputs[name].at(1) + "_" + inputs[name].at(2);
        auto channels = _tensors[inputs[name].at(0)]->dims().at(1);
        auto gamma_dims = memory::dims({channels, channels});
        _tensors[gamma_diff_name] =
            move(make_shared<Tensor>(gamma_diff_name, gamma_dims));
        _tensors[gamma_diff_name]->set_producer(bwd_prefix + name);
      }
      float epsilon = float_parameters["epsilon"];
      float momentum = float_parameters["momentum"];
      _operators.push_back(move(make_shared<BatchNormalization>(
          name, input, output, epsilon, momentum, _tensors, _training)));
    } else if (type == "InstanceNormalization") {
      float epsilon = float_parameters["epsilon"];
      _operators.push_back(move(make_shared<InstanceNormalization>(
          name, input, output, epsilon, _tensors, _training)));
    } else if (type == "Add") {
      _operators.push_back(
          move(make_shared<Add>(name, input, output, _tensors, _training)));
    } else if (type == "Concat") {
      int axis = int_parameters["axis"];
      _operators.push_back(move(
          make_shared<Concat>(name, input, output, axis, _tensors, _training)));
    } else if (type == "Softmax") {
      const auto input_tensor = input[0];
      const auto out_name = output[0];
      _tensors[out_name] =
          move(make_shared<Tensor>(out_name, _tensors[input_tensor]->dims()));
      _tensors[out_name]->set_producer(name);
      auto labels_name = "labels";
      _tensors[labels_name]->add_consumer(fwd_prefix + name);
      auto loss_name = "loss";
      _tensors[loss_name] =
          move(make_shared<Tensor>(loss_name, _tensors[labels_name]->dims()));
      _tensors[loss_name]->set_producer(name);
      _tensors[loss_name]->add_consumer("external");
      if (_training) {
        auto out_diff_name = "diff_" + input_tensor;
        _tensors[out_diff_name] = move(
            make_shared<Tensor>(out_diff_name, _tensors[input_tensor]->dims()));
        _tensors[out_diff_name]->set_producer(name);
      }
      const int axis = _tensors[input_tensor]->dims().size() - 1;
      _operators.push_back(move(make_shared<SoftmaxWithLoss>(
          name, vector<string>({input_tensor, labels_name}),
          vector<string>({out_name, loss_name}), axis, _tensors, _training)));
    } else {
      throw runtime_error("unsupported operator type: " + type);
    }
    for (auto tensor : output) {
      _tensors[tensor]->set_producer(fwd_prefix + name);
    }
    for (auto tensor : input) {
      _tensors[tensor]->add_consumer(fwd_prefix + name);
    }
    if (_training) {
      for (auto tensor : input) {
        auto out_diff_name = "diff_" + tensor;
        _tensors[out_diff_name] =
            move(make_shared<Tensor>(out_diff_name, _tensors[tensor]->dims()));
        _tensors[out_diff_name]->set_producer(bwd_prefix + name);
      }
    }
  }
  if (_operators.at(_operators.size() - 1)->type.find("Softmax") ==
      string::npos) {
    const string name = "softmax0";
    auto out_name = "prediction";
    auto loss_name = "loss";
    auto labels_name = "labels";
    const auto input_tensor =
        _operators.at(_operators.size() - 1)->output.at(0);
    const auto input_dims = _tensors[input_tensor]->dims();
    const int axis = input_dims.size() - 1;
    _tensors[out_name] = move(make_shared<Tensor>(out_name, input_dims));
    _tensors[out_name]->set_producer(name);
    _tensors[out_name]->add_consumer("external");
    _tensors[loss_name] =
        move(make_shared<Tensor>(loss_name, _tensors[labels_name]->dims()));
    _tensors[loss_name]->set_producer(name);
    _tensors[loss_name]->add_consumer("external");
    _tensors[input_tensor]->add_consumer(fwd_prefix + name);
    _tensors[labels_name]->add_consumer(fwd_prefix + name);
    if (_training) {
      auto out_diff_name = "diff_" + input_tensor;
      _tensors[out_diff_name] = move(
          make_shared<Tensor>(out_diff_name, _tensors[input_tensor]->dims()));
      _tensors[out_diff_name]->set_producer(name);
    }
    _operators.push_back(move(make_shared<SoftmaxWithLoss>(
        name, vector<string>({input_tensor, labels_name}),
        vector<string>({out_name, loss_name}), axis, _tensors, _training)));
  }
}

void Network::_fillModelParameters(onnx::ModelProto &model) {
  const auto initializers = model.graph().initializer();
  for (const auto &onnx_info : initializers) {
    if (onnx_info.data_type() != 1) {
      cout << onnx_info.name() << " has unsupported data type "
           << onnx_info.data_type() << "!" << endl;
      continue;
    };
    const auto name = onnx_info.name();
    const auto dims = _tensors[name]->dims();
    const auto desc =
        memory::desc({dims, g_data_type, _get_tag(0, dims.size(), 1)});
    float *raw_data;
    auto v = vector<float>();
    if (onnx_info.raw_data().size() == desc.get_size()) {
      raw_data = reinterpret_cast<float *>(
          const_cast<char *>(onnx_info.raw_data().data()));
    } else if (static_cast<size_t>(onnx_info.float_data().size()) ==
               static_cast<size_t>(desc.get_size() / sizeof(float))) {
      for (auto y : onnx_info.float_data()) {
        v.push_back(static_cast<float>(y));
      }
      raw_data = v.data();
    } else {
      throw runtime_error("Could not initialize " + name);
    }
    _tensors[name]->release();
    auto cpu_device = getCPUDevice();
    _tensors[name]->set_device(cpu_device);
    _tensors[name]->set_producer("external");
    _tensors[name]->set_memory(move(make_memory(
        desc, cpu_device->get_engine(), static_cast<void *>(raw_data))));
  }
}

void Network::_fillInputTensors(const string &data_path,
                                const string &label_path, const size_t &batch) {
  const auto data_tensor_name = _operators.at(0)->input[0];
  const auto labels_name = "labels";
  auto in_dims = _tensors[data_tensor_name]->dims();
  auto in_desc =
      memory::desc({in_dims, g_data_type, _get_tag(1, in_dims.size(), 0)});
  auto v_in = vector<float>(in_desc.get_size() / sizeof(float));
  auto l_dims = _tensors[labels_name]->dims();
  auto l_desc =
      memory::desc({l_dims, g_data_type, _get_tag(0, l_dims.size(), 0)});
  auto v_l = vector<float>(l_desc.get_size() / sizeof(float));
  if (data_path != "") {
    const auto batchsize = in_dims.at(0);
    const int writeImages = 0;
    load_data_to_handle<uint8_t>(v_in.data(), batch, batchsize, data_path,
                                 writeImages);
    if (data_path.find("imagenet") != string::npos) {
      const auto channels = in_dims.at(1);
      size_t size = 1;
      for (size_t i = 2; i < in_dims.size(); i++) {
        size *= in_dims.at(i);
      };
      normalizeImages(batchsize, channels, size, v_in.data());
      load_data_to_handle<int16_t>(v_l.data(), batch, l_dims.at(0), label_path);
    } else {
      load_data_to_handle<uint8_t>(v_l.data(), batch, l_dims.at(0), label_path);
    }
  } else {
    generate(v_in.begin(), v_in.end(), _rand_float);
    generate(v_l.begin(), v_l.end(), _rand_float);
  }
  float *buffer = v_in.data();
  _tensors[data_tensor_name]->release();
  auto cpu_device = getCPUDevice();
  _tensors[data_tensor_name]->set_device(cpu_device);
  _tensors[data_tensor_name]->set_memory(move(make_memory(
      in_desc, cpu_device->get_engine(), static_cast<void *>(buffer))));
  _tensors[labels_name]->release();
  _tensors[labels_name]->set_device(cpu_device);
  _tensors[labels_name]->set_memory(move(make_memory(
      l_desc, cpu_device->get_engine(), static_cast<void *>(v_l.data()))));
}

void Network::_print_memory_usage(const string memory_file = "",
                                  const string event_info = "") {
  string memory_usage = "";
  for (auto dev : _devices) {
    memory_usage += to_string(dev.second->memory_used) + ",";
  }
  memory_usage += event_info;
  if (!memory_file.empty()) {
    string cmd = "echo " + memory_usage + " >> " + memory_file;
    system(cmd.c_str());
  } else {
    cout << memory_usage << endl;
  }
}

/**************************************************************/

float runOP(int is_fwd_pass, shared_ptr<Operator> &op, shared_ptr<Device> &dev,
            unordered_map<std::string, shared_ptr<Tensor>> &tensors,
            memory::format_tag out_tag, int verbose) {
  size_t num_executions = 50;
  dnnl_set_verbose(0);
  auto begin = get_time();
  int measure_time = 0;
  auto runtimes = vector<float>();
  for (size_t i = 0; i < num_executions; i++) {
    if (i == num_executions - 1) {
      measure_time = 1;
      dnnl_set_verbose(verbose > 1);
    }
    begin = get_time();
    if (is_fwd_pass) {
      op->forward(dev, tensors, out_tag, measure_time);
    } else {
      op->backward(dev, tensors, out_tag, measure_time);
    }
    runtimes.push_back(get_elapsed_ms(begin));
  }
  dnnl_set_verbose(0);
  vector<float>::iterator median_time = runtimes.begin();
  std::advance(median_time, runtimes.size() / 2);
  std::nth_element(runtimes.begin(), median_time, runtimes.end());
  return *median_time;
}

float Network::_computeOp(const size_t computeSchedID, const string devName) {
  auto outTag = memory::format_tag::any;
  auto is_fwd_pass = computeSchedID < _operators.size();
  string mode = is_fwd_pass ? "fwd" : "bwd";
  size_t computeOpID =
      is_fwd_pass ? computeSchedID : 2 * _operators.size() - computeSchedID - 1;
  string computeOpName = _operators.at(computeOpID)->name;
  string computeOpType = _operators.at(computeOpID)->type;
  if (_verbose > 1) {
    cout << "compute " << computeSchedID << " " << computeOpName << " ("
         << computeOpType << ")"
         << " " << mode << " on device " << devName << endl;
  }
  auto inputs = is_fwd_pass ? _operators.at(computeOpID)->_f_op.input
                            : _operators.at(computeOpID)->_b_op.input;
  _reinitTensors(inputs);
  float median_time = 0.0f;
  packaged_task<float(int, shared_ptr<Operator> &, shared_ptr<Device> &,
                      unordered_map<std::string, shared_ptr<Tensor>> &,
                      memory::format_tag, int)>
      task(runOP);
  auto future = task.get_future();
  thread thr(move(task), is_fwd_pass, std::ref(_operators.at(computeOpID)),
             std::ref(_devices[devName]), std::ref(_tensors), outTag, _verbose);
  if (future.wait_for(500s) != future_status::timeout) {
    thr.join();
    if (future.valid()) {
      median_time = future.get();
      // opTime:   time in ms of Xth (last) operator execution
      float opTime = _getTimeOfOp(computeOpID, mode + "_", "total");
      if (_verbose > 1) {
        cout << computeOpType << ": " << opTime << " vs. " << median_time
             << endl;
      }
    }
  } else {
    thr.detach();
    throw std::runtime_error("Timeout in operator " + computeOpName + "_" +
                             mode + "!");
  }
  return median_time;
}

void Network::_releaseOp(const size_t releaseSchedID) {
  size_t releaseOpID = (releaseSchedID < _operators.size())
                           ? releaseSchedID
                           : 2 * _operators.size() - releaseSchedID - 1;
  if (_verbose > 1) {
    cout << "free " << to_string(releaseSchedID) << " (OpID "
         << to_string(releaseOpID) << ")" << endl;
  }
  if (releaseSchedID < _operators.size()) {
    _releaseTensors(_operators.at(releaseOpID)->_f_op.output);
    _operators.at(releaseOpID)->reset_fwd_primitives();
  } else {
    _releaseTensors(_operators.at(releaseOpID)->_b_op.output);
    _operators.at(releaseOpID)->reset_bwd_primitives();
    _releaseTensors(_operators.at(releaseOpID)->_f_op.output);
    _operators.at(releaseOpID)->reset_fwd_primitives();
  }
}

vector<float> Network::_run(const string &data_path, const string &label_path,
                            const size_t num_iterations) {
  _fillInputTensors(data_path, label_path, 0);
  for (auto t = _tensors.begin(); t != _tensors.end(); t++) {
    if (t->second->producer() == "external") {
      auto consumers = t->second->consumers();
      if (consumers.size() > 0) {
        auto firstConsumerName = consumers[0];
        auto consumerSchedID =
            getOpIndexFromName(_operators, firstConsumerName);
        if (_operators.at(consumerSchedID)->type == "BatchNormalization") {
          auto t_name = t->second->name();
          if (t_name.find("gamma") != string::npos ||
              t_name.find("beta") != string::npos) {
            if (_verbose > 1) {
              cout << "skip " << t->second->name() << " (BN param)" << endl;
            }
            continue;
          }
        }
        string devName = _getExecuteOperator(consumerSchedID).engineID;
        if (_verbose > 1) {
          cout << "move model parameter " << t->second->name() << " to "
               << devName << endl;
        }
        t->second->init(t->second->desc(), _devices[devName]);
      }
    }
  }
  _print_memory_usage(_memoryLogfile, "loaded_params");

  auto opTimes = vector<float>();

  if (_R.get_size() > 0) {
    if (_verbose > 1) {
      cout << "R" << endl;
      _R.print();
      cout << "S" << endl;
      _S.print();
      cout << "F" << endl;
      _F.print();
    }
    size_t T = _R.get_rows();
    for (auto i = 0; i < num_iterations; i++) {
      for (size_t t = 0; t < T; t++) {
        // Release
        for (size_t n = 0; n < t; n++) {
          if ((_S.at(0, t, n) + _S.at(1, t, n)) == 0) {
            _releaseOp(n);
          }
        }
        // Compute
        for (size_t n = 0; n <= t; n++) {
          if (_R.at(0, t, n) == 1) {
            opTimes.push_back(_computeOp(n, "cpu_0"));
          }
          if (_R.at(1, t, n) == 1) {
            opTimes.push_back(_computeOp(n, "gpu_0"));
          }
        }
        // measure memory usage after each timestep
        _print_memory_usage(_memoryLogfile, "t" + to_string(t));
      }
    }
  } else {
    size_t T = _training ? 2 * _operators.size() : _operators.size();
    for (auto i = 0; i < num_iterations; i++) {
      for (size_t t = 0; t < T; t++) {
        // Release
        if (t > _opsToKeep) {
          size_t releaseSchedID = t - 2;
          _releaseOp(releaseSchedID);
        }
        // Compute
        string devName = _getExecuteOperator(t).engineID;
        opTimes.push_back(_computeOp(t, devName));
        // measure memory usage after each timestep
        _print_memory_usage(_memoryLogfile, "t" + to_string(t));
      }
    }
  }

  float total_time = accumulate(opTimes.begin(), opTimes.end(), 0.0);
  if (_verbose > 0) {
    printf("run took %0.2lf ms.\n", total_time);
  }
  if (_verbose > 1) {
    cout << "median times: " << endl;
    for (size_t opID = 0; opID < _operators.size(); opID++) {
      cout << "   " << _operators.at(opID)->name << ": " << opTimes[opID]
           << endl;
    }
  }

  _resetPrimitives();
  for (auto t = _tensors.begin(); t != _tensors.end(); t++) {
    t->second->release();
  }
  _print_memory_usage(_memoryLogfile, "finished_run");
  return opTimes;
}

vector<vector<float>> Network::_benchmark(const string &data_path,
                                          const string &label_path) {
  vector<vector<float>> compute_costs;
  // measure performance on all devices
  for (auto dev = _devices.begin(); dev != _devices.end(); dev++) {
    string dev_name = dev->first;
    if (_verbose > 0) {
      cout << "*** " << dev_name << " *** " << _mode << " ***" << endl;
    }
    vector<vector<string>> sched = _createScheduleStringVec(dev_name);
    _setSchedule(sched);
    vector<float> opTimes = _run(data_path, label_path, 1);
    compute_costs.push_back(opTimes);
  }
  return compute_costs;
}

void Network::_reinitTensors(vector<string> &tensor_names) {
  for (auto i : tensor_names) {
    if (!_tensors[i]->is_initialized()) {
      if (_verbose > 1) {
        cout << "reinit " << _tensors[i]->name() << endl;
      }
      _tensors[i]->reinit(_tensors[i]->desc());
    }
  }
}

void Network::_releaseTensors(vector<string> &tensor_names) {
  for (auto t : tensor_names) {
    if (_tensors[t]->is_initialized()) {
      if (_verbose > 1) {
        cout << "release " << _tensors[t]->name() << endl;
      }
      _tensors[t]->release();
    }
  }
}

void Network::_resetPrimitives() {
  for (auto op : _operators) {
    op->reset_fwd_primitives();
    op->reset_bwd_primitives();
  }
}

/**************************************************************/

vector<vector<string>>
Network::_createScheduleStringVec(vector<string> &device_per_op) {
  auto sched = vector<vector<string>>();
  for (size_t schedID = 0; schedID < _operators.size(); schedID++) {
    vector<string> dec = {_operators.at(schedID)->type, device_per_op[schedID],
                          "0", "any"};
    sched.push_back(dec);
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      auto opID = _operators.size() - i - 1;
      auto schedID = _operators.size() + i;
      vector<string> dec = {_operators.at(opID)->type, device_per_op[schedID],
                            "0", "any"};
      sched.push_back(dec);
    }
  }
  return sched;
}

vector<vector<string>>
Network::_createScheduleStringVec(const string device_name) {
  const int num_ops = _training ? 2 * _operators.size() : _operators.size();
  auto device_per_op = vector<string>(num_ops);
  fill(device_per_op.begin(), device_per_op.end(), device_name);
  return _createScheduleStringVec(device_per_op);
}

void Network::_setSchedule(const string &schedulefile) {
  _unsetSchedule();
  _schedule = move(make_unique<Schedule>(schedulefile));
}

void Network::_setSchedule(vector<vector<string>> &sched) {
  _unsetSchedule();
  _schedule = move(make_unique<Schedule>(sched));
}

void Network::_unsetSchedule() {
  if (_schedule != nullptr) {
    _schedule.reset();
    _schedule.release();
    _schedule = nullptr;
  }
}

/**************************************************************/

void Network::_scheduleOperatorMinTime(const size_t &opID, const string prefix,
                                       string &best_schedule) {
  auto time_per_op = map<string, float>();
  for (auto device = _devices.begin(); device != _devices.end(); device++) {
    auto dev_name = device->first;
    time_per_op[dev_name] = _getTimeOfOp(opID, prefix, "total");
  }
  pair<string, float> best =
      *min_element(time_per_op.begin(), time_per_op.end(),
                   [](pair<string, float> i, pair<string, float> j) {
                     return i.second < j.second;
                   });
  best_schedule += _operators.at(opID)->type + ";" + best.first + ";0;any\n";
  if (_verbose > 1) {
    cout << "best choice: " + best.first << endl;
  }
}

void Network::_writeScheduleFileMinTime(const string &schedulefile) {
  string best_schedule = "";
  for (size_t opID = 0; opID < _operators.size(); opID++) {
    _scheduleOperatorMinTime(opID, "fwd_", best_schedule);
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      auto opID = _operators.size() - i - 1;
      auto schedID = _operators.size() + i;
      _scheduleOperatorMinTime(opID, "bwd_", best_schedule);
    }
  }
  if (_verbose > 0) {
    cout << "Best schedule: \n" << best_schedule << endl;
  }
  writeString2File(schedulefile, best_schedule);
}

/**************************************************************/

ExecuteOperator Network::_getExecuteOperator(const int ID) {
  if (_schedule == nullptr)
    throw runtime_error("No schedule defined!");
  if (_schedule->empty())
    throw runtime_error("Schedule is empty!");
  auto d = _schedule->get(ID);
  if (d->type != DecisionType::PRODUCE_TENSOR) {
    throw runtime_error("Unsupported DecisionType!");
  }
  ExecuteOperator *e = static_cast<ExecuteOperator *>(d);
  return *e;
}

float Network::_getTimeOfOp(const int opID, const string prefix,
                            const string time_type) {
  auto op = _operators.at(opID);
  auto timings = op->timings;
  auto e = _getExecuteOperator(opID);
  auto time_name = prefix + e.engineID;
  float time = timings[time_name][time_type];
  if (_verbose > 1) {
    string s = op->name + "\n: ";
    for (auto &it : timings[time_name]) {
      s += to_string(it.second) + " ms (" + it.first + ")  ";
    }
    s += "\n";
    cout << s << endl;
  }
  return time;
}

void Network::_ilpMatrices2Schedule(const string &schedulefile) {
  // This only computes a very dumb schedule (only frontier advancing stage, no
  // recomputes)
  string best_schedule = "";
  for (size_t opID = 0; opID < _operators.size(); opID++) {
    string device_name = _R.at(0, opID, opID) == 1 ? "cpu_0" : "gpu_0";
    best_schedule += _operators.at(opID)->type + ";" + device_name + ";0;any\n";
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      auto opID = _operators.size() - i - 1;
      auto schedID = _operators.size() + i;
      string device_name = _R.at(0, schedID, schedID) == 1 ? "cpu_0" : "gpu_0";
      best_schedule +=
          _operators.at(opID)->type + ";" + device_name + ";0;any\n";
    }
  }
  if (_verbose > 0) {
    cout << "Best schedule: \n" << best_schedule << endl;
  }
  writeString2File(schedulefile, best_schedule);
}

/**************************************************************/
float Network::_getTensorCopyCosts(string tensor_name, string src_dev_name,
                                   string dst_dev_name) {
  auto d_src = getDevIndexFromName(_devices, src_dev_name);
  auto d_dst = getDevIndexFromName(_devices, dst_dev_name);
  if (d_src == d_dst) {
    // same device, no copy costs
    return 0.0;
  }
  auto src_eng = _devices[src_dev_name]->get_engine();
  auto dst_eng = _devices[dst_dev_name]->get_engine();

  // TODO: handle different src and target desc!
  auto src_desc = _tensors[tensor_name]->desc();
  auto dst_desc = _tensors[tensor_name]->desc();

  auto s = stream(dst_eng);
  if (src_eng.get_kind() != engine::kind::cpu) {
    s = stream(src_eng);
  }
  auto begin = get_time();
  auto src = make_memory(src_desc, src_eng);
  auto dst = make_memory(dst_desc, dst_eng);
  reorder(src, dst).execute(s, {{DNNL_ARG_FROM, src}, {DNNL_ARG_TO, dst}});
  s.wait();
  return get_elapsed_ms(begin);
}

void Network::_fillCopyCostsMatrix(matrix &copy_costs,
                                   vector<pair<string, edge>> &edges) {
  for (auto e : edges) {
    const string tensor_name = e.first;
    auto edgeID =
        getEdgeIndexFromSrcDst(edges, e.second.get_u(), e.second.get_v());

    for (auto d = _devices.begin(); d != _devices.end(); d++) {
      auto src_dev_name = d->first;
      auto d_src = getDevIndexFromName(_devices, src_dev_name);

      for (auto d_ = _devices.begin(); d_ != _devices.end(); d_++) {
        auto dst_dev_name = d_->first;
        auto d_dst = getDevIndexFromName(_devices, dst_dev_name);

        float c = _getTensorCopyCosts(tensor_name, src_dev_name, dst_dev_name);
        copy_costs.set(edgeID, d_dst, d_src, c);
      }
    }
  }
}
/**************************************************************/