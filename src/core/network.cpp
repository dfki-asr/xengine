#include "network.hpp"
#ifdef HAS_CBC
#include "ilp_solver_cbc.cpp"
#endif
#ifdef HAS_GUROBI
#include "ilp_solver_grb.cpp"
#endif

using namespace std;
using namespace dnnl;

Network::Network(const string model_name, const string &model_path,
                 const string &devices_path, const int training,
                 const int verbose)
    : _model_name(model_name), _training(training), _schedule(nullptr),
      _verbose(verbose), _measure_time(0), _benchmark_mode(0) {
  _mode = _training ? "training" : "inference";
  create_devices(_devices, devices_path);
  _default_device = _devices.begin()->first;
  _model = load_model(model_path);
  _tensors = get_tensors(_model);
  unordered_map<string, vector<string>> inputs, outputs;
  _preprocessModel(inputs, outputs);
  _initOperators(inputs, outputs);
  _opsToKeep = 1;
  if (_verbose > 0) {
    cout << endl
         << "********** " << _model_name << " *** " << _mode << " ********"
         << endl;
    maxMemoryDemandInfo();
  }
}

Network::~Network() {
  _primitives.clear();
  _primitive_args.clear();
  for (auto t = _tensors.begin(); t != _tensors.end(); t++) {
    t->second->release();
    t->second.reset();
    t->second.release();
  }
  _tensors.clear();
  for (auto op = _operators.begin(); op != _operators.end(); op++) {
    op->reset();
    op->release();
  }
  _operators.clear();
  for (auto dev = _devices.begin(); dev != _devices.end(); dev++) {
    dev->second.reset();
  }
  _devices.clear();
  _operator_names.clear();
  unsetSchedule();
}

void Network::init() {
  auto begin = get_time();
  _fillModelParameters();
  const auto data_tensor_name = _model.graph().input()[0].name();
  const auto labels_name = "labels";
  _tensors[data_tensor_name]->producer = "external";
  _tensors[labels_name]->producer = "external";
  if (_verbose > 0) {
    cout << "init took " << (get_elapsed_ms(begin)) << " ms." << endl;
  }
}

void Network::maxMemoryDemandInfo() {
  float op_md = 0.0f;
  for (size_t opID = 0; opID < _operators.size(); opID++) {
    op_md += _operators.at(opID)->getFwdMemoryConsumption();
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      auto opID = _operators.size() - i - 1;
      op_md += _operators.at(opID)->getBwdMemoryConsumption();
    }
  }
  op_md /= (1024.0f * 1024.0f);
  float tensor_md = 0.0f;
  for (auto it = _tensors.begin(); it != _tensors.end(); it++) {
    string t_name = it->first;
    float bytes = product(it->second->dims()) * sizeof(float);
    if (_verbose > 1) {
      cout << t_name << " with " << to_string(bytes / (1024.0f * 1024.0f))
           << " MB." << endl;
    }
    tensor_md += bytes;
  }
  tensor_md /= (1024.0f * 1024.0f);
  cout << "max memory if keeping all tensors at the same time: "
       << to_string(tensor_md) << " MB." << endl;
}

void Network::setSchedule(const string &schedule_file) {
  unsetSchedule();
  _schedule = move(make_unique<Schedule>(schedule_file));
}

void Network::unsetSchedule() {
  if (_schedule != nullptr) {
    _schedule.reset();
    _schedule.release();
    _schedule = nullptr;
  }
}

void Network::run(const string &data_path, const string &label_path,
                  const size_t num_iterations) {
  for (auto i = 0; i < num_iterations; i++) {
    _fillInputTensors(data_path, label_path, i);
    _forward();
    if (_training) {
      _backward();
    }
  }
  _reset_op_primitives();
}

void Network::_reset_op_primitives() {
  for (auto i = 0; i < _operators.size(); i++) {
    _operators.at(i)->reset_fwd_primitives();
    _operators.at(i)->reset_bwd_primitives();
  }
}

void Network::benchmark(const string &data_path, const string &label_path) {
  // be sure to ignore any schedule
  unsetSchedule();
  _benchmark_mode = 1;
  // measure performance on all devices
  for (auto dev = _devices.begin(); dev != _devices.end(); dev++) {
    _default_device = dev->first;
    if (_verbose > 0) {
      cout << "*** " << _default_device << " *** " << _mode << " ***" << endl;
    }
    run(data_path, label_path, 1);
  }
  _benchmark_mode = 0;
}

void Network::writeScheduleFile(const string &schedule_file) {
  string best_schedule;
  for (size_t opID = 0; opID < _operators.size(); opID++) {
    _scheduleOperator(opID, "fwd_", best_schedule);
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      auto opID = _operators.size() - i - 1;
      auto schedID = _operators.size() + i;
      _scheduleOperator(opID, "bwd_", best_schedule);
    }
  }
  if (_verbose > 0) {
    cout << "Best schedule: \n" << best_schedule << endl;
  }
  writeString2File(schedule_file, best_schedule);
}

vector<string> Network::_selectDevicePerOp(vector<string> dev_names,
                                           const int srcDifferent) {
  auto device_per_op = vector<string>();
  for (size_t i = 0; i < _operators.size(); i++) {
    size_t dev_idx = (i % dev_names.size());
    auto dev_name = dev_names[dev_idx];
    device_per_op.push_back(dev_name);
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      size_t schedID = _operators.size() + i;
      size_t dev_idx = (schedID % dev_names.size());
      string dev_name = dev_names[dev_idx];
      if (srcDifferent) {
        int opID = 2 * _operators.size() - schedID - 2;
        if (opID >= 0) {
          if (dev_name == device_per_op[opID]) {
            size_t idx = dev_idx == 1 ? 0 : 1;
            dev_name = dev_names[idx];
          }
        }
      }
      device_per_op.push_back(dev_name);
    }
  }
  return device_per_op;
}

vector<size_t> Network::_get_uncovered_edges(vector<pair<string, edge>> &edges,
                                             matrix &copy_costs) {
  vector<size_t> edges_uncovered;
  // Go through all copy costs and check if we forgot something
  for (size_t i = 0; i < edges.size(); i++) {
    float tmp = 0;
    for (size_t d_ = 0; d_ < _devices.size(); d_++) {
      for (size_t d = 0; d < _devices.size(); d++) {
        // get copy costs from d -> to d_
        tmp += copy_costs.at(i, d, d_);
      }
    }
    if (tmp > 0.0f) {
      continue;
    } else {
      edges_uncovered.push_back(i);
    }
  }
  return edges_uncovered;
}

void Network::_collectConsumerCopyCosts(const int opID, const int d,
                                        vector<string> outputs,
                                        vector<string> &device_per_op,
                                        vector<pair<string, edge>> &edges,
                                        matrix &copy_costs) {
  for (auto output : outputs) {
    for (auto consumer : _tensors[output]->consumers) {
      if (consumer == "external") {
        continue;
      }
      auto consumerSchedID = _getOpIndexFromName(consumer);
      auto cons_dev_name = device_per_op[consumerSchedID];
      auto d_ = _getDevIndexFromName(cons_dev_name);
      if (d == d_) {
        continue;
      }
      auto consumerOpID = consumerSchedID;
      string prefix = "fwd_";
      if (consumerSchedID >= _operators.size()) {
        consumerOpID = 2 * _operators.size() - consumerSchedID - 1;
        prefix = "bwd_";
      }
      auto timings = _operators.at(consumerOpID)->timings;
      auto producer = _tensors[output]->producer;
      const size_t src_idx = _getOpIndexFromName(producer);
      const size_t dst_idx = consumerSchedID;
      const string edgeName = to_string(src_idx) + "->" + to_string(dst_idx);
      float edgeCosts = timings[prefix + cons_dev_name][output];
      auto edgeID = getEdgeIndexFromName(edges, edgeName);
      // set copy costs from d -> to d_
      float c = copy_costs.at(edgeID, d, d_);
      if (c == 0.0f) {
        copy_costs.set(edgeID, d, d_, edgeCosts);
      }
    }
  }
}

void Network::_fillCopyCosts(matrix &copy_costs, vector<string> &device_per_op,
                             vector<pair<string, edge>> &edges) {
  // Execute full graph on different devices
  string schedule_file = "tmp_schedule.txt";
  string sched;
  for (size_t opID = 0; opID < _operators.size(); opID++) {
    sched += _operators.at(opID)->type + ";" + device_per_op[opID] + ";0;any\n";
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      auto opID = _operators.size() - i - 1;
      auto schedID = _operators.size() + i;
      sched +=
          _operators.at(opID)->type + ";" + device_per_op[schedID] + ";0;any\n";
    }
  }
  writeString2File(schedule_file, sched);
  setSchedule(schedule_file);
  // Execute
  _forward();
  if (_training) {
    _backward();
  }
  // Collect costs
  for (size_t opID = 0; opID < _operators.size(); opID++) {
    auto d = _getDevIndexFromName(device_per_op[opID]);
    _collectConsumerCopyCosts(opID, d, _operators.at(opID)->_f_op.output,
                              device_per_op, edges, copy_costs);
  }
  if (_training) {
    for (size_t i = 0; i < _operators.size(); i++) {
      auto opID = _operators.size() - i - 1;
      auto schedID = _operators.size() + i;
      auto d = _getDevIndexFromName(device_per_op[schedID]);
      _collectConsumerCopyCosts(schedID, d, _operators.at(opID)->_b_op.output,
                                device_per_op, edges, copy_costs);
    }
  }
}

void Network::solveILP(const string mpsfile, const string logfile,
                       vector<pair<string, edge>> &edges,
                       vector<string> &dev_names,
                       vector<vector<float>> &compute_costs_per_op,
                       vector<float> &memory_per_op, matrix &copy_costs,
                       vector<float> &budget, vector<float> &ram) {
  unique_ptr<ILP_Solver> ilp = nullptr;
  // choose Gurobi if available, otherwise CBC
#ifdef HAS_GUROBI
  ilp = make_unique<ILP_Solver_GRB>(
      _model_name + "_" + _mode, mpsfile, logfile, edges, dev_names,
      compute_costs_per_op, memory_per_op, copy_costs, budget, ram, _verbose);
#else
#ifdef HAS_CBC
  ilp = make_unique<ILP_Solver_CBC>(
      _model_name + "_" + _mode, mpsfile, logfile, edges, dev_names,
      compute_costs_per_op, memory_per_op, copy_costs, budget, ram, _verbose);
#endif
#endif
  if (ilp == nullptr) {
    throw runtime_error("ILP Solver error!");
  }
  cout << "solve ILP ..." << endl;
  ilp->solve();
  ilp->printResults();
  matrix R = ilp->get_R();
  matrix S = ilp->get_S();
  vector<double> results;
  results.push_back(ilp->get_minimal_compute_costs());
  results.push_back(ilp->get_minimal_memory());
  results.push_back(ilp->get_maximal_memory());
}

void Network::solveILP(const string mpsfile, const string logfile,
                       const string &data_path, const string &label_path,
                       const int benchmarkILP) {
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
    auto src = it->second->producer;
    auto consumers = it->second->consumers;
    for (auto dst : consumers) {
      if (src == "external" || dst == "external") {
        continue;
      }
      if (src == dst) {
        continue;
      }
      const size_t src_idx = _getOpIndexFromName(src);
      const size_t dst_idx = _getOpIndexFromName(dst);
      const string edgeName = to_string(src_idx) + "->" + to_string(dst_idx);
      edges.push_back(pair<string, edge>(edgeName, edge(src_idx, dst_idx)));
    }
  }
  matrix copy_costs = matrix(edges.size(), _devices.size(), _devices.size(), 0);

  if (benchmarkILP) {
    if (_verbose > 0) {
      cout << "Benchmark ILP, aquire data ..." << endl;
    }
    benchmark("", "");
    // compute costs
    auto time_type = "total";
    for (size_t d = 0; d < dev_names.size(); d++) {
      _default_device = dev_names[d];
      compute_costs_per_op.push_back(vector<float>());
      for (size_t opID = 0; opID < _operators.size(); opID++) {
        compute_costs_per_op[d].push_back(
            _getTimeOfOp(opID, "fwd_", time_type));
      }
      if (_training) {
        for (size_t i = 0; i < _operators.size(); i++) {
          auto opID = _operators.size() - i - 1;
          compute_costs_per_op[d].push_back(
              _getTimeOfOp(opID, "bwd_", time_type));
        }
      }
    }
    // memory costs
    for (size_t opID = 0; opID < _operators.size(); opID++) {
      memory_per_op.push_back(_operators.at(opID)->getFwdMemoryConsumption());
    }
    if (_training) {
      for (size_t i = 0; i < _operators.size(); i++) {
        auto opID = _operators.size() - i - 1;
        memory_per_op.push_back(_operators.at(opID)->getBwdMemoryConsumption());
      }
    }
    if (_devices.size() > 1) {
      cout << "measure copy costs ..." << endl;
      vector<string> device_per_op;
      do {
        _reset_op_primitives();
        device_per_op = _selectDevicePerOp(dev_names);
        _fillCopyCosts(copy_costs, device_per_op, edges);
      } while (next_permutation(dev_names.begin(), dev_names.end()));
      if (_training) {
        do {
          _reset_op_primitives();
          device_per_op = _selectDevicePerOp(dev_names, 1);
          _fillCopyCosts(copy_costs, device_per_op, edges);
        } while (next_permutation(dev_names.begin(), dev_names.end()));
      }
      _reset_op_primitives();
      vector<size_t> edges_uncovered = _get_uncovered_edges(edges, copy_costs);
      for (size_t d = 0; d < _devices.size(); d++) {
        vector<string> cover_last_edges_device_per_op = device_per_op;
        for (size_t i : edges_uncovered) {
          size_t src = edges[i].second.get_u();
          size_t dst = edges[i].second.get_v();
          auto op2switch = (d == 0) ? dst : src;
          cover_last_edges_device_per_op[op2switch] =
              (device_per_op[op2switch] == dev_names[0]) ? dev_names[1]
                                                         : dev_names[0];
        }
        _reset_op_primitives();
        _fillCopyCosts(copy_costs, cover_last_edges_device_per_op, edges);
      }
      vector<size_t> edges_still_uncovered =
          _get_uncovered_edges(edges, copy_costs);
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
      cout << _model_name << " " << _mode
           << " has computational costs:" << endl;
      for (size_t d = 0; d < dev_names.size(); d++) {
        cout << dev_names[d] << ": ";
        for (auto c : compute_costs_per_op[d]) {
          printf("%0.3lf ms  ", c);
        }
        cout << endl;
      }
      cout << _model_name << " " << _mode << " has memory costs:" << endl;
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
      {"250MB_250MB", {268435456, 268435456}},
      {"150MB_150MB", {157286400, 157286400}},
      {"100MB_100MB", {104857600, 104857600}}};
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

ExecuteOperator Network::_getExecuteOperator(const int ID) {
  if (_schedule != nullptr) {
    if (_schedule->size() != 0) {
      auto d = _schedule->get(ID);
      if (d->type != DecisionType::PRODUCE_TENSOR) {
        throw runtime_error("Unsupported DecisionType!");
      };
      ExecuteOperator *e = static_cast<ExecuteOperator *>(d);
      return *e;
    }
  }
  return ExecuteOperator(to_string(ID), _default_device, 0, "any");
}

void Network::_maybe_provide_dummy_inputs(vector<string> &inputs) {
  if (_benchmark_mode) {
    for (auto i : inputs) {
      if (!_tensors[i]->is_initialized()) {
        if (_verbose > 1) {
          cout << "reinit " << _tensors[i]->name << endl;
        }
        _tensors[i]->reinit(_tensors[i]->desc());
      }
    }
  }
}

void Network::_maybe_release_outputs(vector<string> &outputs) {
  if (_benchmark_mode) {
    for (auto o : outputs) {
      _tensors[o]->release();
    }
  }
}

void Network::_maybe_release_op(const int opID, const int schedID) {
  if (_benchmark_mode) {
    if (opID < _operators.size()) {
      _operators.at(opID)->reset_fwd_primitives();
    } else {
      _operators.at(opID)->reset_bwd_primitives();
    }
  }
}

void Network::_forward() {
  const int is_fwd_pass = 1;
  _Xpass(is_fwd_pass);
}

void Network::_backward() {
  const int is_fwd_pass = 0;
  _Xpass(is_fwd_pass);
}

void Network::_Xpass(const int is_fwd_pass) {
  size_t num_executions = 10;
  size_t warmup_iterations = 5;
  vector<float> avg_times = vector<float>();
  size_t opID, schedID;
  int releaseOpID, releaseSchedID;
  vector<string> inputs, outputs;
  string mode;
  for (size_t j = 0; j < _operators.size(); j++) {
    if (is_fwd_pass) {
      opID = j;
      schedID = j;
      inputs = _operators.at(opID)->_f_op.input;
      mode = "fwd";
    } else {
      opID = _operators.size() - j - 1;
      schedID = _operators.size() + j;
      inputs = _operators.at(opID)->_b_op.input;
      mode = "bwd";
    }
    if (_verbose > 1) {
      cout << "compute " << to_string(schedID) << " "
           << _operators.at(opID)->name << " " << mode << endl;
    }
    // Release
    if (schedID > _opsToKeep) {
      if (is_fwd_pass) {
        releaseOpID = opID - 2;
        releaseSchedID = releaseOpID;
        outputs = _operators.at(releaseOpID)->_f_op.output;
        if (_verbose > 1) {
          cout << "free " << to_string(releaseOpID) << " fwd" << endl;
        }
        _maybe_release_outputs(outputs);
        _maybe_release_op(releaseOpID, releaseSchedID);
      } else {
        releaseSchedID = schedID - 2;
        if (releaseSchedID >= _operators.size()) {
          releaseOpID = 2 * _operators.size() - releaseSchedID - 1;
          if (_verbose > 1) {
            cout << "free " << to_string(releaseOpID) << " bwd" << endl;
          }
          outputs = _operators.at(releaseOpID)->_b_op.output;
          _maybe_release_outputs(outputs);
          _maybe_release_op(releaseOpID, releaseSchedID);
          if (_verbose > 1) {
            cout << "free " << to_string(releaseOpID) << " fwd" << endl;
          }
          outputs = _operators.at(releaseOpID)->_f_op.output;
          _maybe_release_outputs(outputs);
          _maybe_release_op(releaseOpID, releaseSchedID);
        } else {
          releaseOpID = releaseSchedID;
          if (_verbose > 1) {
            cout << "free " << to_string(releaseOpID) << " fwd" << endl;
          }
          outputs = _operators.at(releaseOpID)->_f_op.output;
          _maybe_release_outputs(outputs);
          _maybe_release_op(releaseOpID, releaseSchedID);
        }
      }
    }
    // Compute
    _maybe_provide_dummy_inputs(inputs);
    auto e = _getExecuteOperator(schedID);
    auto eng = _devices[e.engineID]->get_engine();
    auto out_tag = e.outputTag.to_dnnl();
    _measure_time = 0;
    dnnl_set_verbose(0);
    for (size_t i = 0; i < warmup_iterations; i++) {
      if (is_fwd_pass) {
        _operators.at(opID)->forward(*_devices[e.engineID].get(), _tensors,
                                     out_tag, _measure_time);
      } else {
        _operators.at(opID)->backward(*_devices[e.engineID].get(), _tensors,
                                      out_tag, _measure_time);
      }
    }
    auto begin = get_time();
    for (size_t i = 0; i < num_executions; i++) {
      if (i == num_executions - 1) {
        _measure_time = 1;
        dnnl_set_verbose(_verbose > 1);
      }
      if (is_fwd_pass) {
        _operators.at(opID)->forward(*_devices[e.engineID].get(), _tensors,
                                     out_tag, _measure_time);
      } else {
        _operators.at(opID)->backward(*_devices[e.engineID].get(), _tensors,
                                      out_tag, _measure_time);
      }
    }
    dnnl_set_verbose(0);
    avg_times.push_back(get_elapsed_ms(begin) /
                        static_cast<float>(num_executions));
  }
  float total_avg = accumulate(avg_times.begin(), avg_times.end(), 0.0);
  if (_verbose > 0) {
    printf("%s took %0.2lf ms in average.\n", mode.c_str(), total_avg);
  }
  if (_verbose > 1) {
    cout << "avg times: " << endl;
    for (size_t opID = 0; opID < _operators.size(); opID++) {
      cout << "   " << mode << " " << _operators.at(opID)->name << ": "
           << avg_times[opID] << endl;
    }
  }
}

void Network::_insertSoftmax() {
  const string name = "softmax0";
  auto out_name = "prediction";
  auto loss_name = "loss";
  auto labels_name = "labels";
  const auto input_tensor = _operators.at(_operators.size() - 1)->output.at(0);
  const auto input_dims = _tensors[input_tensor]->dims();
  const int axis = input_dims.size() - 1;
  _tensors[out_name] = move(make_unique<Tensor>(out_name, input_dims));
  _tensors[out_name]->producer = name;
  _tensors[out_name]->add_consumer("external");
  _tensors[loss_name] =
      move(make_unique<Tensor>(loss_name, _tensors[labels_name]->dims()));
  _tensors[loss_name]->producer = name;
  _tensors[loss_name]->add_consumer("external");
  _tensors[input_tensor]->add_consumer("fwd_" + name);
  _tensors[labels_name]->add_consumer("fwd_" + name);
  if (_training) {
    auto out_diff_name = "diff_" + input_tensor;
    _tensors[out_diff_name] = move(
        make_unique<Tensor>(out_diff_name, _tensors[input_tensor]->dims()));
    _tensors[out_diff_name]->producer = name;
  }
  _operators.push_back(move(make_unique<SoftmaxWithLoss>(
      name, vector<string>({input_tensor, labels_name}),
      vector<string>({out_name, loss_name}), axis, _tensors, _training)));
  _operator_names.push_back(name);
}

void Network::_preprocessModel(unordered_map<string, vector<string>> &inputs,
                               unordered_map<string, vector<string>> &outputs) {
  const auto nodes = _model.graph().node();
  for (const auto &node : nodes) {
    const auto name = node.name();
    inputs[name] = get_string_vector_from_proto(node.input());
    outputs[name] = get_string_vector_from_proto(node.output());
  }
  // remove Flatten operators, remove Dropout operators in non-training mode
  // mark as to-be-replaced
  unordered_map<string, int> replace_info;
  for (auto i = 0; i < nodes.size(); i++) {
    const auto type = nodes.at(i).op_type();
    replace_info[nodes.at(i).name()] =
        (type == "Flatten" || (type == "Dropout" && _training == false));
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
    auto org_prev_output = outputs[prev].at(0);
    if (inputs[next].size() > 1) {
      auto w_name = inputs[next].at(1);
      auto w_dims = _tensors[w_name]->dims();
      auto d_dims = _tensors[org_prev_output]->dims();
      auto new_dims = vector<memory::dim>();
      new_dims.push_back(w_dims.at(0));
      for (size_t j = 1; j < d_dims.size(); j++)
        new_dims.push_back(d_dims.at(j));
      _tensors[w_name]->set_dims(memory::dims(new_dims));
    }
    outputs[prev].at(0) = inputs[next].at(0);
    inputs[next].at(0) = outputs[prev].at(0);
    const auto name = nodes.at(i).name();
    inputs[name] = vector<string>();
    outputs[name] = vector<string>();
  }
  const string fwd_prefix = "fwd_";
  const string bwd_prefix = "bwd_";
  for (const auto &node : nodes) {
    const auto name = node.name();
    const auto input = inputs[name];
    const auto output = outputs[name];
    for (auto tensor : output)
      _tensors[tensor]->producer = fwd_prefix + name;
    for (auto tensor : input)
      _tensors[tensor]->add_consumer(fwd_prefix + name);
  }
  if (_training) {
    for (const auto &node : nodes) {
      const auto name = node.name();
      const auto type = node.op_type();
      if (type.find("Pool") != string::npos) {
        auto output_name = outputs[name].at(0);
        auto ws_name = output_name + "_ws";
        _tensors[ws_name] =
            move(make_unique<Tensor>(ws_name, _tensors[output_name]->dims()));
        _tensors[ws_name]->producer = fwd_prefix + name;
      } else if (type == "BatchNormalization") {
        auto gamma_diff_name =
            "diff_" + inputs[name].at(1) + "_" + inputs[name].at(2);
        auto channels = _tensors[inputs[name].at(0)]->dims().at(1);
        auto gamma_dims = memory::dims({channels, channels});
        _tensors[gamma_diff_name] =
            move(make_unique<Tensor>(gamma_diff_name, gamma_dims));
        _tensors[gamma_diff_name]->producer = bwd_prefix + name;
      }
      for (auto tensor : inputs[name]) {
        auto out_diff_name = "diff_" + tensor;
        _tensors[out_diff_name] =
            move(make_unique<Tensor>(out_diff_name, _tensors[tensor]->dims()));
        _tensors[out_diff_name]->producer = bwd_prefix + name;
      }
    }
  }
}

void Network::_initOperators(unordered_map<string, vector<string>> &inputs,
                             unordered_map<string, vector<string>> &outputs) {
  for (const auto &node : _model.graph().node()) {
    const auto name = node.name();
    const auto type = node.op_type();
    const auto input = inputs[name];
    const auto output = outputs[name];
    unordered_map<string, vector<memory::dim>> dim_parameters;
    unordered_map<string, float> float_parameters;
    unordered_map<string, int> int_parameters;
    get_params_from_proto(node, dim_parameters, float_parameters,
                          int_parameters);
    if (type == "Flatten" || (type == "Dropout" && _training == false))
      continue;
    if (type == "Conv") {
      _operators.push_back(
          make_unique<Conv>(name, input, output, dim_parameters["strides"],
                            dim_parameters["kernel_shape"],
                            dim_parameters["pads"], _tensors, _training));
    } else if (type == "ConvTranspose") {
      _operators.push_back(move(make_unique<ConvTranspose>(
          name, input, output, dim_parameters["strides"],
          dim_parameters["kernel_shape"], dim_parameters["pads"], _tensors,
          _training)));
    } else if (type == "MaxPool") {
      _operators.push_back(move(
          make_unique<MaxPool>(name, input, output, dim_parameters["strides"],
                               dim_parameters["kernel_shape"],
                               dim_parameters["pads"], _tensors, _training)));
    } else if (type == "AveragePool") {
      _operators.push_back(move(make_unique<AveragePool>(
          name, input, output, dim_parameters["strides"],
          dim_parameters["kernel_shape"], dim_parameters["pads"], _tensors,
          _training)));
    } else if (type == "GlobalAveragePool") {
      const auto i_dims = _tensors[input.at(0)]->dims();
      memory::dims kernel = {i_dims.at(2), i_dims.at(3)};
      _operators.push_back(move(make_unique<GlobalAveragePool>(
          name, input, output, kernel, _tensors, _training)));
    } else if (type == "Gemm") {
      _operators.push_back(
          move(make_unique<Gemm>(name, input, output, _tensors, _training)));
    } else if (type == "Relu") {
      _operators.push_back(
          move(make_unique<Relu>(name, input, output, _tensors, _training)));
    } else if (type == "LeakyRelu") {
      float alpha = float_parameters["alpha"];
      _operators.push_back(move(make_unique<LeakyRelu>(
          name, input, output, alpha, _tensors, _training)));
    } else if (type == "Dropout") {
      float probability = float_parameters["ratio"];
      _operators.push_back(move(make_unique<Dropout>(
          name, input, output, probability, _tensors, _training)));
    } else if (type == "BatchNormalization") {
      float epsilon = float_parameters["epsilon"];
      float momentum = float_parameters["momentum"];
      _operators.push_back(move(make_unique<BatchNormalization>(
          name, input, output, epsilon, momentum, _tensors, _training)));
    } else if (type == "InstanceNormalization") {
      float epsilon = float_parameters["epsilon"];
      _operators.push_back(move(make_unique<InstanceNormalization>(
          name, input, output, epsilon, _tensors, _training)));
    } else if (type == "Add") {
      _operators.push_back(
          move(make_unique<Add>(name, input, output, _tensors, _training)));
    } else if (type == "Concat") {
      int axis = int_parameters["axis"];
      _operators.push_back(move(
          make_unique<Concat>(name, input, output, axis, _tensors, _training)));
    } else {
      throw runtime_error("unsupported operator type: " + type);
    }
    _operator_names.push_back(name);
  }
  _insertSoftmax();
}

void Network::_fillModelParameters() {
  const auto initializers = _model.graph().initializer();
  for (const auto &onnx_info : initializers) {
    if (onnx_info.data_type() != 1) {
      throw runtime_error("Unsupported ONNX data type!");
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
    auto tensor_memory_cpu = make_memory(desc, _devices["cpu_0"]->get_engine(),
                                         static_cast<void *>(raw_data));
    _tensors[name]->set_memory(tensor_memory_cpu);
    _tensors[name]->producer = "external";
  }
}

void Network::_fillInputTensors(const string &data_path,
                                const string &label_path, const size_t &batch) {
  const auto data_tensor_name = _model.graph().input()[0].name();
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
  auto tensor_in_cpu = make_memory(in_desc, _devices["cpu_0"]->get_engine(),
                                   static_cast<void *>(buffer));
  auto tensor_l_cpu = make_memory(l_desc, _devices["cpu_0"]->get_engine(),
                                  static_cast<void *>(v_l.data()));
  _tensors[data_tensor_name]->release();
  _tensors[data_tensor_name]->set_memory(move(tensor_in_cpu));
  _tensors[labels_name]->release();
  _tensors[labels_name]->set_memory(move(tensor_l_cpu));
}

float Network::_getTimeOfOp(const int opID, const string prefix,
                            const string time_type) {
  auto timings = _operators.at(opID)->timings;
  auto e = _getExecuteOperator(opID);
  auto time_name = prefix + e.engineID;
  float time = timings[time_name][time_type];
  if (_verbose > 1) {
    string s = _operators.at(opID)->name + "\n: ";
    for (auto &it : timings[time_name]) {
      s += to_string(it.second) + " ms (" + it.first + ")  ";
    }
    s += "\n";
    cout << s << endl;
  }
  return time;
}

void Network::_scheduleOperator(const size_t &opID, const string prefix,
                                string &best_schedule) {
  map<string, float> time_per_op;
  for (auto device = _devices.begin(); device != _devices.end(); device++) {
    _default_device = device->first;
    time_per_op[_default_device] = _getTimeOfOp(opID, prefix, "total");
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

int Network::_getOpIndexFromName(const string opName) {
  for (size_t opID = 0; opID < _operators.size(); opID++) {
    if ("fwd_" + _operators[opID]->name == opName) {
      return opID;
    }
  }
  for (size_t i = 0; i < _operators.size(); i++) {
    auto opID = _operators.size() - i - 1;
    if ("bwd_" + _operators[opID]->name == opName) {
      return _operators.size() + i;
    }
  }
  throw runtime_error("Operator " + opName + " was not found!");
}

int Network::_getDevIndexFromName(const string devName) {
  size_t idx = 0;
  for (auto dev = _devices.begin(); dev != _devices.end(); dev++) {
    if (dev->first == devName) {
      return idx;
    }
    idx += 1;
  }
  throw runtime_error("Device " + devName + " was not found!");
}