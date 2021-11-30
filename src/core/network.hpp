#ifndef XENGINE_NETWORK_HPP
#define XENGINE_NETWORK_HPP

#include "../onnx_utils.hpp"
#include "device.hpp"
#include "ilp_solver.hpp"
#include "schedule.hpp"

using namespace std;
using namespace dnnl;

class Network {
public:
  Network(const string model_name, const string &model_path,
          const string &devices_path, const int training,
          const int verbose = 0);
  ~Network();
  void init();
  void run(const string &data_path, const string &label_path,
           const size_t num_iterations);
  void benchmark(const string &data_path, const string &label_path);
  void writeScheduleFile(const string &schedule_file);
  void setSchedule(const string &schedule_file);
  void unsetSchedule();
  void solveILP(const string mpsfile, const string logfile,
                vector<pair<string, edge>> &edges, vector<string> &dev_names,
                vector<vector<float>> &compute_costs_per_op,
                vector<float> &memory_per_op, matrix &copy_costs,
                vector<float> &budget, vector<float> &ram);
  void solveILP(const string mpsfile, const string logfile,
                const string &data_path, const string &label_path,
                const int benchmarkILP = 1);
  void solveILP(const string mpsfile, const string logfile);
  void maxMemoryDemandInfo();

private:
  void _Xpass(const int is_fwd_pass);
  void _forward();
  void _backward();
  void _preprocessModel(unordered_map<string, vector<string>> &inputs,
                        unordered_map<string, vector<string>> &outputs);
  void _initOperators(unordered_map<string, vector<string>> &inputs,
                      unordered_map<string, vector<string>> &outputs);
  void _insertSoftmax();
  void _fillModelParameters();
  void _fillInputTensors(const string &data_path, const string &label_path,
                         const size_t &batch);
  ExecuteOperator _getExecuteOperator(const int ID);
  void _scheduleOperator(const size_t &opID, const string prefix,
                         string &best_schedule);
  float _getTimeOfOp(const int opID, const string prefix,
                     const string time_type);
  int _getOpIndexFromName(const string opName);
  int _getDevIndexFromName(const string devName);
  vector<string> _selectDevicePerOp(vector<string> dev_names,
                                    const int srcDifferent = 0);
  vector<size_t> _get_uncovered_edges(vector<pair<string, edge>> &edges,
                                      matrix &copy_costs);
  void _fillCopyCosts(matrix &copy_costs, vector<string> &device_per_op,
                      vector<pair<string, edge>> &edges);
  void _collectConsumerCopyCosts(const int opID, const int d,
                                 vector<string> outputs,
                                 vector<string> &device_per_op,
                                 vector<pair<string, edge>> &edges,
                                 matrix &copy_costs);
  void _maybe_provide_dummy_inputs(vector<string> &inputs);
  void _maybe_release_outputs(vector<string> &outputs);
  void _maybe_release_op(const int opID, const int schedID);
  void _reset_op_primitives();

  onnx::ModelProto _model;
  vector<string> _operator_names;
  map<string, shared_ptr<Device>> _devices;
  unordered_map<string, unique_ptr<Tensor>> _tensors;
  vector<unique_ptr<Operator>> _operators;
  vector<unique_ptr<primitive>> _primitives;
  vector<unordered_map<int, memory>> _primitive_args;
  unique_ptr<Schedule> _schedule;
  string _default_device;
  int _measure_time;
  int _verbose;
  int _training;
  int _benchmark_mode;
  int _opsToKeep;
  string _mode;
  string _model_name;
};
#endif