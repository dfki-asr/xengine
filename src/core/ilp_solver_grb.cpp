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

GRBLinExpr ILP_Solver_GRB::_num_hazards(const size_t t, const size_t d,
                                        const size_t i, const size_t k,
                                        GRBVar *R, GRBVar *S) {
  const size_t T = _compute_costs[d].size();
  GRBLinExpr sum_succ = 0;
  for (auto j : _successors(i)) {
    if (j > k) {
      sum_succ += R[(d * T * T) + t * T + j];
    }
  }
  GRBLinExpr num_hazards = 1 - R[(d * T * T) + t * T + k] + sum_succ;
  if (t + 1 < T) {
    num_hazards += S[(d * T * T) + (t + 1) * T + i];
  }
  return num_hazards;
}

int ILP_Solver_GRB::defineProblem() {
  try {
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", _logfile);
    env.set("LogToConsole", to_string(_verbose > 0));
    env.start();

    GRBModel model = GRBModel(env);
    model.set(GRB_StringAttr_ModelName, _model_name);
    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model.set(GRB_DoubleParam_TimeLimit, 1200);

    const size_t T = _compute_costs[0].size();
    const size_t E = _edges.size();
    const size_t D = _devices.size();

    // create variables
    size_t t, i, d, e, k, e_idx, s;
    GRBVar R[D * T * T], S[D * T * T], Z[D * T * T], U[D * T * T], F[D * T * E];
    for (d = 0; d < D; ++d) {
      auto gcd = _budget[d] / _ram[d];
      for (t = 0; t < T; ++t) {
        for (i = 0; i < T; ++i) {
          const size_t idx = (d * T * T) + (t * T) + i;
          const string idx_str = idx_to_string(d, t, i);
          R[idx] = model.addVar(0, 1, 0, GRB_BINARY, "R" + idx_str);
          S[idx] = model.addVar(0, 1, 0, GRB_BINARY, "S" + idx_str);
          Z[idx] = model.addVar(0, 1, 0, GRB_BINARY, "Z" + idx_str);
          U[idx] = model.addVar(0, gcd, 0, GRB_CONTINUOUS, "U" + idx_str);
        }
        for (e = 0; e < E; ++e) {
          const size_t idx = (d * T * E) + (t * E) + e;
          F[idx] =
              model.addVar(0, 1, 0, GRB_BINARY, "F" + idx_to_string(d, t, e));
        }
      }
    }
    GRBQuadExpr obj = 0;
    for (d = 0; d < D; ++d) {
      for (t = 0; t < T; ++t) {
        for (i = 0; i < T; ++i) {
          const size_t idx = (d * T * T) + (t * T) + i;
          obj += R[idx] * _compute_costs[d][i];
        }
      }
    }
    // copies between devices
    for (t = 0; t < T; ++t) {
      for (auto e : _edges) {
        for (auto d_ = 0; d_ < D; ++d_) {
          // consumer
          const size_t idx_v = (d_ * T * T) + (t * T) + e.second.get_v();
          for (d = 0; d < D; ++d) {
            // producer
            const size_t idx_u = (d * T * T) + (t * T) + e.second.get_u();
            const string edgeName = e.first;
            const size_t edgeIdx = getEdgeIndexFromName(_edges, edgeName);
            // get copy costs from d -> to d_
            const float cc = _copy_costs.at(edgeIdx, d, d_);
            if (cc > 0) {
              obj += R[idx_v] * cc * Z[idx_u];
            }
          }
        }
      }
    }
    model.setObjective(obj, GRB_MINIMIZE);

    // add Constraints

    // Z is a helper variable that is 1 if either R or S is 1 and 0 otherwise
    for (t = 0; t < T; ++t) {
      for (i = 0; i < T; ++i) {
        for (d = 0; d < D; ++d) {
          const size_t idx = (d * T * T) + (t * T) + i;
          model.addConstr(Z[idx], GRB_LESS_EQUAL, R[idx] + S[idx]);
          model.addConstr(Z[idx], GRB_GREATER_EQUAL, R[idx]);
          model.addConstr(Z[idx], GRB_GREATER_EQUAL, S[idx]);
        }
      }
    }
    // upper right part of R (excl. diagonal) is 0
    GRBLinExpr R_upperRight = 0;
    for (d = 0; d < D; ++d) {
      for (t = 0; t < T; ++t) {
        for (i = t + 1; i < T; ++i) {
          const size_t idx = (d * T * T) + (t * T) + i;
          R_upperRight += R[idx];
        }
      }
    }
    model.addConstr(R_upperRight, GRB_EQUAL, 0);
    // exactly one new evaluation per timestep
    GRBLinExpr R_newEval = 0;
    for (t = 0; t < T; ++t) {
      GRBLinExpr R_tmp;
      for (d = 0; d < D; ++d) {
        const size_t idx = (d * T * T) + (t * T) + t;
        R_tmp += R[idx];
      }
      R_newEval += R_tmp;
      // at least one evaluation on any device
      model.addConstr(R_tmp, GRB_GREATER_EQUAL, 1);
    }
    model.addConstr(R_newEval, GRB_EQUAL, T);
    // upper right part of S (incl. diagonal) is 0
    GRBLinExpr S_upperRight = 0;
    for (d = 0; d < D; ++d) {
      for (t = 0; t < T; ++t) {
        for (i = t; i < T; ++i) {
          const size_t idx = (d * T * T) + (t * T) + i;
          S_upperRight += S[idx];
        }
      }
    }
    model.addConstr(S_upperRight, GRB_EQUAL, 0);
    // ensure all checkpoints are in memory
    for (t = 0; t < T - 1; ++t) {
      for (i = 0; i < T; ++i) {
        for (d = 0; d < D; ++d) {
          const size_t idx = (d * T * T) + (t * T) + i;
          const size_t idx_next = (d * T * T) + ((t + 1) * T) + i;
          model.addConstr(S[idx_next], GRB_LESS_EQUAL, S[idx] + R[idx]);
        }
      }
    }
    // ensure all computations are possible
    for (t = 0; t < T; ++t) {
      for (auto e : _edges) {
        for (d = 0; d < D; ++d) {
          GRBLinExpr S_idx_u = 0;
          GRBLinExpr R_idx_u = 0;
          for (auto d_ = 0; d_ < D; ++d_) {
            const size_t idx_u = (d_ * T * T) + (t * T) + e.second.get_u();
            S_idx_u += S[idx_u];
            R_idx_u += R[idx_u];
          }
          const size_t idx_v = (d * T * T) + (t * T) + e.second.get_v();
          model.addConstr(R[idx_v], GRB_LESS_EQUAL, R_idx_u + S_idx_u);
        }
      }
    }
    // upper and lower bounds for 1 - F
    for (t = 0; t < T; ++t) {
      for (e_idx = 0; e_idx < E; ++e_idx) {
        GRBLinExpr F_linExp = 0;
        GRBLinExpr F_haz = 0;
        GRBLinExpr F_haz_max = 0;
        for (d = 0; d < D; ++d) {
          edge e = _edges[e_idx].second;
          const size_t i = e.get_u();
          const size_t k = e.get_v();
          F_linExp += (1 - F[((d * T * E) + t * E) + e_idx]);
          F_haz += _num_hazards(t, d, i, k, R, S);
          F_haz_max += _max_num_hazards(t, d, i, k) *
                       (1 - F[(d * T * E) + t * E + e_idx]);
        }
        // upper bound
        model.addConstr(F_linExp, GRB_LESS_EQUAL, F_haz);
        // lower bound
        model.addConstr(F_haz_max, GRB_GREATER_EQUAL, F_haz);
      }
    }
    // initialize memory usage (includes spurious checkpoints)
    for (t = 0; t < T; ++t) {
      GRBLinExpr lhs = 0;
      GRBLinExpr rhs = 0;
      for (d = 0; d < D; ++d) {
        GRBLinExpr cp = 0;
        for (i = 0; i < T; ++i) {
          cp += (S[(d * T * T) + t * T + i] * _memory_costs[i]);
        }
        lhs += U[(d * T * T) + t * T];
        rhs += R[(d * T * T) + t * T] * _memory_costs[0] + cp;
      }
      model.addConstr(lhs, GRB_EQUAL, rhs);
    }
    // memory recurrence
    for (t = 0; t < T; ++t) {
      for (k = 0; k < T - 1; ++k) {
        for (d = 0; d < D; ++d) {
          vector<pair<size_t, size_t>> pred = _predecessors(k);
          GRBLinExpr mem_freed = 0;
          for (auto p : pred) {
            const size_t e_idx = p.second;
            mem_freed += _memory_costs[i] * F[(d * T * E) + t * E + e_idx];
          }
          GRBLinExpr lhs = U[(d * T * T) + t * T + (k + 1)];
          GRBLinExpr rhs =
              U[(d * T * T) + t * T + k] +
              R[(d * T * T) + t * T + (k + 1)] * _memory_costs[k + 1] -
              mem_freed;
          model.addConstr(lhs, GRB_EQUAL, rhs);
        }
      }
    }
    // memory budget constraints
    for (d = 0; d < D; ++d) {
      auto gcd = _budget[d] / _ram[d];
      for (t = 0; t < T * T; ++t) {
        model.addConstr(U[(d * T * T) + t], GRB_GREATER_EQUAL, 0);
        model.addConstr(U[(d * T * T) + t], GRB_LESS_EQUAL, gcd);
      }
    }
    // write model to file
    if (_verbose > 0) {
      cout << "Write ilp model file " << _mpsfile << endl;
    }
    model.write(_mpsfile);
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

int ILP_Solver_GRB::solve() {
  ifstream f(_mpsfile);
  if (!f.is_open()) {
    defineProblem();
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