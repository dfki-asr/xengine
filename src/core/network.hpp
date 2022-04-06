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
  shared_ptr<Device> getCPUDevice() {
    if (_cpu_device != nullptr) {
      return _cpu_device;
    }
  }
  void createSchedule(const string &schedule_file, const string &images,
                      const string &labels);
  void runSchedule(const string &schedule_file, const string &images,
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
  void _preprocessModel(onnx::ModelProto &model,
                        unordered_map<string, vector<string>> &inputs,
                        unordered_map<string, vector<string>> &outputs);
  void _init(onnx::ModelProto &model,
             unordered_map<string, vector<string>> &inputs,
             unordered_map<string, vector<string>> &outputs);
  void _fillModelParameters(onnx::ModelProto &model);
  void _fillInputTensors(const string &data_path, const string &label_path,
                         const size_t &batch);
  void _print_memory_usage(const string memory_file, const string event_info);
  /**************************************************************/
  void _releaseOp(const size_t releaseSchedID);
  float _computeOp(const size_t computeSchedID, const string devName);
  vector<float> _run(const string &data_path, const string &label_path,
                     const size_t num_iterations);
  vector<vector<float>> _benchmark(const string &data_path,
                                   const string &label_path);
  void _reinitTensors(vector<string> &tensor_names);
  void _releaseTensors(vector<string> &tensor_names);
  void _resetPrimitives();
  /**************************************************************/
  vector<vector<string>> _createScheduleStringVec(const string device_name);
  vector<vector<string>>
  _createScheduleStringVec(vector<string> &device_per_op);
  void _setSchedule(vector<vector<string>> &sched);
  void _setSchedule(const string &schedulefile);
  void _unsetSchedule();
  /**************************************************************/
  void _scheduleOperatorMinTime(const size_t &opID, const string prefix,
                                string &best_schedule);
  void _writeScheduleFileMinTime(const string &schedulefile);
  /**************************************************************/
  ExecuteOperator _getExecuteOperator(const int ID);
  float _getTimeOfOp(const int opID, const string prefix,
                     const string time_type);
  void _ilpMatrices2Schedule(const string &schedulefile);
  /**************************************************************/
  void _fillCopyCosts(matrix &copy_costs, vector<string> &device_per_op,
                      vector<pair<string, edge>> &edges);
  void _collectConsumerCopyCosts(const int opID, const int d,
                                 vector<string> outputs,
                                 vector<string> &device_per_op,
                                 vector<pair<string, edge>> &edges,
                                 matrix &copy_costs);
  /**************************************************************/
  string _name, _mode, _output_dir, _memoryLogfile;
  map<string, shared_ptr<Device>> _devices;
  shared_ptr<Device> _cpu_device;
  unordered_map<string, shared_ptr<Tensor>> _tensors;
  vector<shared_ptr<Operator>> _operators;
  vector<unique_ptr<primitive>> _primitives;
  vector<unordered_map<int, memory>> _primitive_args;
  unique_ptr<Schedule> _schedule;
  matrix _R, _S, _F;
  int _training;
  int _verbose, _measure_time, _opsToKeep;
};
#endif