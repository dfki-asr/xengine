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
  Network(const string name, const string model_file, const string device_file,
          const int training, const string output_dir, const int verbose = 0);
  ~Network();

  string name() { return _name; }
  string mode() { return _mode; }
  string output_directory() { return _output_dir; }
  void run(const string &data_path, const string &label_path,
           const size_t num_iterations);
  void benchmark(const string &data_path, const string &label_path);
  void createSchedule(const string &schedulefile, const string &images,
                      const string &labels);
  void runSchedule(const string &schedulefile, const string &images,
                   const string &labels, const size_t num_iterations);
  void solveILP(const string mpsfile, const string logfile,
                vector<pair<string, edge>> &edges, vector<string> &dev_names,
                vector<vector<float>> &compute_costs_per_op,
                vector<float> &memory_per_op, matrix &copy_costs,
                vector<float> &budget, vector<float> &ram);
  void solveILP(const string mpsfile, const string logfile,
                const string &data_path, const string &label_path,
                const int benchmarkILP = 1);
  void solveILP(const string mpsfile, const string logfile);

private:
  void _Xpass(const int is_fwd_pass);
  void _preprocessModel(onnx::ModelProto &model,
                        unordered_map<string, vector<string>> &inputs,
                        unordered_map<string, vector<string>> &outputs);
  void _init(onnx::ModelProto &model,
             unordered_map<string, vector<string>> &inputs,
             unordered_map<string, vector<string>> &outputs);
  void _fillModelParameters(onnx::ModelProto &model);
  void _fillInputTensors(const string &data_path, const string &label_path,
                         const size_t &batch);
  ExecuteOperator _getExecuteOperator(const int ID);
  void _scheduleOperatorMinTime(const size_t &opID, const string prefix,
                                string &best_schedule);
  float _getTimeOfOp(const int opID, const string prefix,
                     const string time_type);
  void _computeMatrix2Schedule(matrix &R, const string &schedulefile);
  vector<vector<string>> _createScheduleStringVec(string &device_name);
  vector<vector<string>>
  _createScheduleStringVec(vector<string> &device_per_op);
  void _writeScheduleFileMinTime(const string &schedulefile);
  void _setSchedule(vector<vector<string>> &sched);
  void _setSchedule(const string &schedulefile);
  void _unsetSchedule();
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

  map<string, shared_ptr<Device>> _devices;
  unordered_map<string, shared_ptr<Tensor>> _tensors;
  vector<shared_ptr<Operator>> _operators;
  vector<unique_ptr<primitive>> _primitives;
  vector<unordered_map<int, memory>> _primitive_args;
  unique_ptr<Schedule> _schedule;
  int _measure_time;
  int _verbose;
  int _training;
  int _benchmark_mode;
  int _opsToKeep;
  string _mode;
  string _name;
  string _output_dir;
};
#endif