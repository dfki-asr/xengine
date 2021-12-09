#ifndef XENGINE_ILP_SOLVER_GRB_CPP
#define XENGINE_ILP_SOLVER_GRB_CPP

#include "ilp_solver_grb.hpp"
#include "ilp_solver.cpp"

ILP_Solver_GRB::ILP_Solver_GRB(string model_name, string mpsfile,
                               string logfile, vector<pair<string, edge>> edges,
                               vector<string> devices,
                               vector<vector<float>> compute_costs,
                               vector<float> memory_costs, matrix &copy_costs,
                               vector<float> budget, vector<float> ram,
                               const int verbose)
    : ILP_Solver(model_name, mpsfile, logfile, edges, devices, compute_costs,
                 memory_costs, copy_costs, budget, ram, verbose) {
  _mpsfile = mpsfile.empty() ? _model_name + "_grb.mps" : mpsfile;
  _logfile = logfile.empty() ? _model_name + "_grb.log" : logfile;
}
ILP_Solver_GRB::~ILP_Solver_GRB() {}

int ILP_Solver_GRB::solve() {
  ifstream f(_mpsfile);
  if (!f.is_open()) {
    defineProblemAsMPS();
  }
  try {
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", _logfile);
    env.set("LogToConsole", to_string(_verbose > 0));
    env.start();

    if (_verbose > 0) {
      cout << "Read ilp model file " << _mpsfile << endl;
    }
    GRBModel model = GRBModel(env, _mpsfile);

    model.optimize();

    size_t t, i, d, e, k, e_idx, s;
    const size_t T = _compute_costs[0].size();
    const size_t E = _edges.size();
    const size_t D = _devices.size();

    // get times for run on 1 device only
    cout << "******************************************" << endl;
    GRBLinExpr obj = model.getObjective().getLinExpr();
    for (d = 0; d < D; ++d) {
      double device_time = 0;
      for (i = 0; i < T; ++i) {
        const size_t idx = (d * T * T) + (i * T) + i;
        device_time += obj.getCoeff(idx);
      }
      cout << "time on device " << d << ": " << device_time << " ms." << endl;
    }
    cout << "*****************" << endl;

    // get solution(s)
    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
      size_t nSolutionsTotal = model.get(GRB_IntAttr_SolCount);
      size_t nSolutions = 1;
      for (s = 0; s < nSolutions; ++s) {
        model.set(GRB_IntParam_SolutionNumber, s);
        _objective_value = model.get(GRB_DoubleAttr_ObjVal);
        cout << "ILP schedule:     " << _objective_value << " ms." << endl;
        cout << "******************************************" << endl;
        double min_mem = 1e12;
        double max_mem = 0;
        for (d = 0; d < D; ++d) {
          for (t = 0; t < T; ++t) {
            for (i = 0; i < T; ++i) {
              const size_t idx = (d * T * T) + (t * T) + i;
              const string idx_str = idx_to_string(d, t, i);
              _R_matrix->set(
                  idx, model.getVarByName("R" + idx_str).get(GRB_DoubleAttr_X));
              _S_matrix->set(
                  idx, model.getVarByName("S" + idx_str).get(GRB_DoubleAttr_X));
              _Z_matrix->set(
                  idx, model.getVarByName("Z" + idx_str).get(GRB_DoubleAttr_X));
              double u =
                  model.getVarByName("U" + idx_str).get(GRB_DoubleAttr_X);
              _U_matrix->set(idx, u);
              if (u < min_mem) {
                min_mem = u;
              }
              if (u > max_mem) {
                max_mem = u;
              }
            }
            for (e = 0; e < E; ++e) {
              const size_t idx = (d * T * E) + (t * E) + e;
              const string idx_str = idx_to_string(d, t, e);
              _F_matrix->set(
                  idx, model.getVarByName("F" + idx_str).get(GRB_DoubleAttr_X));
            }
          }
        }
        _minimal_memory = min_mem;
        _maximal_memory = max_mem;
      }
    } else {
      throw runtime_error("No solution for " + _model_name);
    }
  } catch (GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
    throw runtime_error("No solution for " + _model_name);
  } catch (...) {
    cout << "Exception during optimization" << endl;
    throw runtime_error("No solution for " + _model_name);
  }
  return 0;
}
#endif