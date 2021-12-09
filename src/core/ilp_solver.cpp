#ifndef XENGINE_ILP_SOLVER_CPP
#define XENGINE_ILP_SOLVER_CPP

#include "ilp_solver.hpp"

string get_fixed_length_var_name(const int c, const int num_chars = 20) {
  string n(num_chars, ' ');
  string s = "R" + to_string(c);
  n.replace(0, s.length(), s);
  return n;
}

string get_fixed_length_var_name(const string s, const int num_chars = 20) {
  string n(num_chars, ' ');
  n.replace(0, s.length(), s);
  return n;
}

string value2str(const float value) {
  stringstream s;
  s << value;
  return s.str();
}

int getEdgeIndexFromName(vector<pair<string, edge>> &edges,
                         const string edgeName) {
  for (size_t edgeID = 0; edgeID < edges.size(); edgeID++) {
    if (edges[edgeID].first == edgeName) {
      return edgeID;
    }
  }
  cout << "Edge " + edgeName + " was not found!" << endl;
  throw runtime_error("Edge " + edgeName + " was not found!");
}

ILP_Solver::ILP_Solver(string model_name, string mpsfile, string logfile,
                       vector<pair<string, edge>> edges, vector<string> devices,
                       vector<vector<float>> compute_costs,
                       vector<float> memory_costs, matrix copy_costs,
                       vector<float> budget, vector<float> ram,
                       const int verbose) {
  _model_name = model_name;
  _mpsfile = mpsfile.empty() ? _model_name + ".mps" : mpsfile;
  _logfile = logfile.empty() ? _model_name + ".log" : logfile;
  _edges = edges;
  _devices = devices;
  _compute_costs = compute_costs;
  _memory_costs = memory_costs;
  _copy_costs = copy_costs;
  _budget = budget;
  _ram = ram;
  _objective_value = -1;
  _minimal_memory = -1;
  _maximal_memory = -1;
  _R_matrix = nullptr;
  _S_matrix = nullptr;
  _Z_matrix = nullptr;
  _U_matrix = nullptr;
  _F_matrix = nullptr;
  _verbose = verbose;
  // init
  const size_t T = _compute_costs[0].size();
  const size_t E = _edges.size();
  const size_t D = _devices.size();
  size_t binary = 1;
  _R_matrix = new matrix(D, T, T, binary);
  _S_matrix = new matrix(D, T, T, binary);
  _Z_matrix = new matrix(D, T, T, binary);
  _F_matrix = new matrix(D, T, E, binary);
  binary = 0;
  _U_matrix = new matrix(D, T, T, binary);

  _rows = "";
  _columns = "";
  _rhs = "";
  _marker = map<string, Marker>();
  _constraints = unordered_map<string, Constraint>();
  _constraints_insertOrder = vector<string>();
  _quad_obj = vector<QuadObj>();
}

ILP_Solver::~ILP_Solver() {
  if (_R_matrix != nullptr) {
    delete _R_matrix;
  }
  if (_S_matrix != nullptr) {
    delete _S_matrix;
  }
  if (_Z_matrix != nullptr) {
    delete _Z_matrix;
  }
  if (_U_matrix != nullptr) {
    delete _U_matrix;
  }
  if (_F_matrix != nullptr) {
    delete _F_matrix;
  }
}

matrix ILP_Solver::get_R() { return *_R_matrix; }
matrix ILP_Solver::get_S() { return *_S_matrix; }
double ILP_Solver::get_minimal_compute_costs() { return _objective_value; }
double ILP_Solver::get_minimal_memory() { return _minimal_memory; }
double ILP_Solver::get_maximal_memory() { return _maximal_memory; }

vector<size_t> ILP_Solver::_successors(const size_t idx) {
  vector<size_t> succ;
  for (auto e : _edges) {
    if (e.second.get_u() == idx) {
      succ.push_back(e.second.get_v());
    }
  }
  return succ;
}

vector<pair<size_t, size_t>> ILP_Solver::_predecessors(const size_t idx) {
  vector<pair<size_t, size_t>> pred;
  for (auto e_idx = 0; e_idx < _edges.size(); ++e_idx) {
    edge e = _edges[e_idx].second;
    if (e.get_v() == idx) {
      pred.push_back(make_pair(e.get_u(), e_idx));
    }
  }
  return pred;
}

size_t ILP_Solver::_max_num_hazards(const size_t t, const size_t d,
                                    const size_t i, const size_t k) {
  const size_t T = _compute_costs[d].size();
  size_t num_uses_after_k = 0;
  for (auto j : _successors(i)) {
    if (j > k) {
      num_uses_after_k += 1;
    }
  }
  if (t + 1 < T) {
    num_uses_after_k += 2;
  } else {
    num_uses_after_k += 1;
  }
  return num_uses_after_k;
}

void ILP_Solver::_num_hazards(const size_t t, const size_t d, const size_t i,
                              const size_t k, const string c_ub,
                              const string c_lb) {
  // ub:    + R[d,t,k] - sum_succ - S[d,t+1,i]   - F[d,t,e_idx]
  // lb: -1 + R[d,t,k] - sum_succ - S[d,t+1,i]   + _max_num_hazards(t,d,i,k)
  //                              - _max_num_hazards(t,d,i,k) * F[d,t,e_idx]
  //  =     + R[d,t,k] - sum_succ - S[d,t+1,i]   - _max_num_hazards(t,d,i,k) *
  //  F[d,t,e_idx]
  //        -1 + _max_num_hazards(t,d,i,k)
  //            --> in rhs as +1 and -_max_num_hazards(t,d,i,k)
  // Note: the +1 is skipped in purpose: it will cancel out for the upper bound
  // and is considered in rhs in lower bound later on
  const size_t T = _compute_costs[d].size();
  size_t sum_succ = 0;
  for (auto j : _successors(i)) {
    if (j > k) {
      // sum_succ += R[d,t,j];
      const string name_F_haz_succ =
          get_fixed_length_var_name("R" + idx_to_string(d, t, j));
      add_marker("B", name_F_haz_succ, c_ub, -1, vector<size_t>{d, t, j});
      add_marker("B", name_F_haz_succ, c_lb, -1, vector<size_t>{d, t, j});
    }
  }
  // num_hazards = + 1 - R[d,t,k] + sum_succ;
  const string name_F_haz_R =
      get_fixed_length_var_name("R" + idx_to_string(d, t, k));
  add_marker("B", name_F_haz_R, c_ub, 1, vector<size_t>{d, t, k});
  add_marker("B", name_F_haz_R, c_lb, 1, vector<size_t>{d, t, k});
  if (t + 1 < T) {
    // num_hazards += S[d,t,i];
    const string name_F_haz_S =
        get_fixed_length_var_name("S" + idx_to_string(d, t + 1, i));
    add_marker("B", name_F_haz_S, c_ub, -1, vector<size_t>{d, t + 1, i});
    add_marker("B", name_F_haz_S, c_lb, -1, vector<size_t>{d, t + 1, i});
  }
}

void ILP_Solver::add_constraint(const string mode, const float rhs) {
  string name = get_fixed_length_var_name(_constraints_insertOrder.size());
  if (_constraints.find(name) == _constraints.end()) {
    _constraints[name] = Constraint(mode, name, rhs);
    _constraints_insertOrder.push_back(name);
  }
}

void ILP_Solver::add_marker(const string mode, const string name,
                            const string type, const float value,
                            const vector<size_t> indices) {
  string content = "    " + name + "  " + type + "  " + to_string(value) + "\n";
  if (_marker.find(name) == _marker.end()) {
    _marker[name] = Marker(mode, name, indices);
  }
  _marker[name].values.push_back(pair<string, float>(type, value));
}

void ILP_Solver::add_row(const string name) {
  auto c = _constraints[name];
  _rows += " " + c.mode + "  " + c.name + "\n";
  if (c.rhs != 0.0) {
    _rhs += "    RHS1      " + c.name + "  " + value2str(c.rhs) + "\n";
  }
}

void ILP_Solver::add_column(const string name) {
  auto m = _marker[name];
  string content = "";
  for (auto p : m.values) {
    string type = p.first;
    content += "    " + name + "  " + type + "  " + value2str(p.second) + "\n";
  }
  if (m.mode == "B") {
    _columns += "    MARKER    'MARKER'                 'INTORG'\n";
    _columns += content;
    _columns += "    MARKER    'MARKER'                 'INTEND'\n";
  } else if (m.mode == "C") {
    _columns += content;
  } else {
    throw runtime_error("Unsupported variable type " + m.mode + "!");
  }
}

int ILP_Solver::defineProblemAsMPS() {
  const size_t T = _compute_costs[0].size();
  const size_t E = _edges.size();
  const size_t D = _devices.size();

  // create variables
  string vars = "";
  size_t t, i, d, e, k, e_idx, s;
  size_t numberOfVariables = 4 * (D * T * T) + (D * T * E);
  for (d = 0; d < D; ++d) {
    auto gcd = _budget[d] / _ram[d];
    for (t = 0; t < T; ++t) {
      for (i = 0; i < T; ++i) {
        const string idx_str = idx_to_string(d, t, i);
        vars += " BV BND1      R" + idx_str + "\n";
        vars += " BV BND1      S" + idx_str + "\n";
        vars += " UP BND1      U" + idx_str + "  " + value2str(gcd) + "\n";
      }
      for (e = 0; e < E; ++e) {
        vars += " BV BND1      F" + idx_to_string(d, t, e) + "\n";
      }
      for (i = 0; i < T; ++i) {
        const string idx_str = idx_to_string(d, t, i);
        vars += " BV BND1      Z" + idx_str + "\n";
      }
    }
  }

  // objective
  const string obj_name = get_fixed_length_var_name("OBJ");
  for (d = 0; d < D; ++d) {
    for (t = 0; t < T; ++t) {
      for (i = 0; i < T; ++i) {
        const string idx_str = idx_to_string(d, t, i);
        const string name = get_fixed_length_var_name("R" + idx_str);
        add_marker("B", name, obj_name, _compute_costs[d][i],
                   vector<size_t>{d, t, i});
      }
    }
  }
  // copies between devices
  // obj += R[idx_v] * _copy_costs.at(e.get_u(), d_, d) *
  //       (R[idx_u] + S[idx_u]);
  for (t = 0; t < T; ++t) {
    for (auto e : _edges) {
      for (auto d_ = 0; d_ < D; ++d_) {
        // const size_t idx_v = (d_ * T * T) + (t * T) + e.get_v();
        for (d = 0; d < D; ++d) {
          // const size_t idx_u = (d * T * T) + (t * T) + e.get_u();
          // consumer
          const string name_R_v = get_fixed_length_var_name(
              "R" + idx_to_string(d_, t, e.second.get_v()));
          // producer
          const string name_R_u = get_fixed_length_var_name(
              "R" + idx_to_string(d, t, e.second.get_u()));
          const string name_S_u = get_fixed_length_var_name(
              "S" + idx_to_string(d, t, e.second.get_u()));
          const string name_Z_u = get_fixed_length_var_name(
              "Z" + idx_to_string(d, t, e.second.get_u()));
          const string edgeName = e.first;
          const size_t edgeIdx = getEdgeIndexFromName(_edges, edgeName);
          // get copy costs from d -> to d_
          const float cc = _copy_costs.at(edgeIdx, d, d_);
          if (cc > 0.0f) {
            _quad_obj.push_back(QuadObj(name_R_v, name_Z_u, cc));
          }
        }
      }
    }
  }

  // constraints

  // Z is a helper variable that is 1 if either R or S is 1 and 0 otherwise
  // model.addConstr(Z[idx], GRB_LESS_EQUAL, R[idx] + S[idx]);
  // model.addConstr(Z[idx], GRB_GREATER_EQUAL, R[idx]);
  // model.addConstr(Z[idx], GRB_GREATER_EQUAL, S[idx]);
  for (t = 0; t < T; ++t) {
    for (i = 0; i < T; ++i) {
      for (d = 0; d < D; ++d) {
        const string idx_str = idx_to_string(d, t, i);
        const string name_Z = get_fixed_length_var_name("Z" + idx_str);
        const string name_R = get_fixed_length_var_name("R" + idx_str);
        const string name_S = get_fixed_length_var_name("S" + idx_str);
        add_constraint("L", 0);
        auto c_Z_sum = _constraints_insertOrder.back();
        add_marker("B", name_R, c_Z_sum, -1, vector<size_t>{d, t, i});
        add_marker("B", name_S, c_Z_sum, -1, vector<size_t>{d, t, i});
        add_marker("B", name_Z, c_Z_sum, 1, vector<size_t>{d, t, i});
        add_constraint("G", 0);
        auto c_Z_R = _constraints_insertOrder.back();
        add_marker("B", name_R, c_Z_R, -1, vector<size_t>{d, t, i});
        add_marker("B", name_Z, c_Z_R, 1, vector<size_t>{d, t, i});
        add_constraint("G", 0);
        auto c_Z_S = _constraints_insertOrder.back();
        add_marker("B", name_S, c_Z_S, -1, vector<size_t>{d, t, i});
        add_marker("B", name_Z, c_Z_S, 1, vector<size_t>{d, t, i});
      }
    }
  }

  // upper right part of R (excl. diagonal) is 0
  // R[0,0,1] + ... = 0
  add_constraint("E", 0);
  auto c_upperRightRZero = _constraints_insertOrder.back();
  for (d = 0; d < D; ++d) {
    for (t = 0; t < T; ++t) {
      for (i = t + 1; i < T; ++i) {
        const string idx_str = idx_to_string(d, t, i);
        const string name = get_fixed_length_var_name("R" + idx_str);
        add_marker("B", name, c_upperRightRZero, 1, vector<size_t>{d, t, i});
      }
    }
  }
  // at least one evaluation on any device
  // R[0,0,0] >= 1
  //    ...   >= 1
  for (t = 0; t < T; ++t) {
    add_constraint("G", 1);
    auto c_oneEvalPerDev = _constraints_insertOrder.back();
    for (d = 0; d < D; ++d) {
      const string idx_str = idx_to_string(d, t, t);
      const string name = get_fixed_length_var_name("R" + idx_str);
      add_marker("B", name, c_oneEvalPerDev, 1, vector<size_t>{d, t, t});
    }
  }
  // exactly one new evaluation per timestep
  // R[0,0,0] + ... = T
  add_constraint("E", T);
  auto c_oneEvalPerStep = _constraints_insertOrder.back();
  for (t = 0; t < T; ++t) {
    for (d = 0; d < D; ++d) {
      const string idx_str = idx_to_string(d, t, t);
      const string name = get_fixed_length_var_name("R" + idx_str);
      add_marker("B", name, c_oneEvalPerStep, 1, vector<size_t>{d, t, t});
    }
  }
  // upper right part of S (incl. diagonal) is 0
  // S[0,0,0] + ... = 0
  add_constraint("E", 0);
  auto c_upperRightSZero = _constraints_insertOrder.back();
  for (d = 0; d < D; ++d) {
    for (t = 0; t < T; ++t) {
      for (i = t; i < T; ++i) {
        const string idx_str = idx_to_string(d, t, i);
        const string name = get_fixed_length_var_name("S" + idx_str);
        add_marker("B", name, c_upperRightSZero, 1, vector<size_t>{d, t, i});
      }
    }
  }
  // ensure all checkpoints are in memory
  // - R[0,0,0] - S[0,0,0] + S[0,1,0] <= 0
  //                ...               <= 0
  // model.addConstr(S[idx_next], GRB_LESS_EQUAL, S[idx] + R[idx]);
  for (t = 0; t < T - 1; ++t) {
    for (i = 0; i < T; ++i) {
      for (d = 0; d < D; ++d) {
        add_constraint("L", 0);
        const string idx_str = idx_to_string(d, t, i);
        const string idx_next_str = idx_to_string(d, t + 1, i);
        const string name_R_idx = get_fixed_length_var_name("R" + idx_str);
        const string name_S_idx = get_fixed_length_var_name("S" + idx_str);
        const string name_S_idx_next =
            get_fixed_length_var_name("S" + idx_next_str);
        auto c_allCP_inMemory = _constraints_insertOrder.back();
        add_marker("B", name_R_idx, c_allCP_inMemory, -1,
                   vector<size_t>{d, t, i});
        add_marker("B", name_S_idx, c_allCP_inMemory, -1,
                   vector<size_t>{d, t, i});
        add_marker("B", name_S_idx_next, c_allCP_inMemory, 1,
                   vector<size_t>{d, t + 1, i});
      }
    }
  }
  // ensure all computations are possible
  // - R[0,0,0] - S[0,0,0] + R[0,0,1] <= 0
  //             ...                  <= 0
  // model.addConstr(R[idx_v], GRB_LESS_EQUAL, R[idx_u] + S_idx_u);
  for (t = 0; t < T; ++t) {
    for (auto e : _edges) {
      for (d = 0; d < D; ++d) {
        add_constraint("L", 0);
        auto c_allCompPossible = _constraints_insertOrder.back();
        for (auto d_ = 0; d_ < D; ++d_) {
          const string idx_u_str_ = idx_to_string(d_, t, e.second.get_u());
          const string name_R = get_fixed_length_var_name("R" + idx_u_str_);
          const string name_S = get_fixed_length_var_name("S" + idx_u_str_);
          add_marker("B", name_R, c_allCompPossible, -1,
                     vector<size_t>{d_, t, e.second.get_u()});
          add_marker("B", name_S, c_allCompPossible, -1,
                     vector<size_t>{d_, t, e.second.get_u()});
        }
        const string idx_u_str = idx_to_string(d, t, e.second.get_u());
        const string idx_v_str = idx_to_string(d, t, e.second.get_v());
        const string name_R_idx_v = get_fixed_length_var_name("R" + idx_v_str);
        add_marker("B", name_R_idx_v, c_allCompPossible, 1,
                   vector<size_t>{d, t, e.second.get_v()});
      }
    }
  }
  // upper and lower bounds for 1 - F
  // R[0,0,1] - F[0,0,0] - S[0,1,0] <= 0
  //              ...
  /*
  // upper bound
  model.addConstr(F_linExp, GRB_LESS_EQUAL, F_haz);
  // - F_haz + F_linExp <= 0
  // - (1 - R[d,t,k] + sum_succ + S[d,t+1,i]) + 1 - F[d,t,e_idx]
  //= -1  + R[d,t,k] - sum_succ - S[d,t+1,i]  + 1 - F[d,t,e_idx]
  //=     + R[d,t,k] - sum_succ - S[d,t+1,i]      - F[d,t,e_idx]
  // ------------------------------------------------
  // lower bound
  model.addConstr(F_haz_max, GRB_GREATER_EQUAL, F_haz);
  // - F_haz + F_haz_max >= 0
  // -1   + R[d,t,k] - sum_succ - S[d,t+1,i]
          + _max_num_hazards(t,d,i,k) * (1 - F[d,t,e_idx])
  // -1   + R[d,t,k] - sum_succ - S[d,t+1,i]
          + _max_num_hazards(t,d,i,k) - _max_num_hazards(t,d,i,k) * F[d,t,e_idx]
  // ------------------------------------------------
  F_linExp  += 1 - F[d,t,e_idx];
  F_haz     += _num_hazards(t,d,i,k);
  F_haz_max += _max_num_hazards(t,d,i,k) * (1 - F[d,t,e_idx]);
  // ------------------------------------------------
  */
  for (t = 0; t < T; ++t) {
    for (e_idx = 0; e_idx < E; ++e_idx) {
      add_constraint("L", 0);
      auto c_F_haz_ub = _constraints_insertOrder.back();
      // init rhs to 0 first until we know _max_num_hazards(t,d,i,k)
      add_constraint("G", 0);
      auto c_F_haz_lb = _constraints_insertOrder.back();
      for (d = 0; d < D; ++d) {
        edge e = _edges[e_idx].second;
        const size_t i = e.get_u();
        const size_t k = e.get_v();
        // upper bound
        // F_linExp += 1 - F[d,t,e_idx]
        const string idx_F_linExp_str = idx_to_string(d, t, e_idx);
        const string name_F_linExp =
            get_fixed_length_var_name("F" + idx_F_linExp_str);
        add_marker("B", name_F_linExp, c_F_haz_ub, -1,
                   vector<size_t>{d, t, e_idx});

        // lower and upper bound
        // F_haz += _num_hazards(t,d,i,k)
        _num_hazards(t, d, i, k, c_F_haz_ub, c_F_haz_lb);

        // lower bound
        // F_haz_max += _max_num_hazards(t,d,i,k) * (1 - F[d,t,e_idx])
        const string name_F_haz_max_F =
            get_fixed_length_var_name("F" + idx_to_string(d, t, e_idx));
        float max_haz = static_cast<float>(_max_num_hazards(t, d, i, k));
        add_marker("B", name_F_haz_max_F, c_F_haz_lb, -max_haz,
                   vector<size_t>{d, t, e_idx});
        // rhs correction due to -1 +_max_num_hazards(t,d,i,k) on lhs!
        _constraints[c_F_haz_lb].rhs += (-max_haz + 1.0);
      }
    }
  }
  // initialize memory usage (includes spurious checkpoints)
  // lhs - rhs == 0
  // - 1e+06 R[0,0,0] - 1e+06 S[0,0,0] + U[0,0,0] - 4e+06 S[0,0,1] = 0
  //                     ...                                       = 0
  // model.addConstr(lhs, GRB_EQUAL, rhs);
  for (t = 0; t < T; ++t) {
    add_constraint("E", 0);
    auto c_initMemory = _constraints_insertOrder.back();
    for (d = 0; d < D; ++d) {
      for (i = 0; i < T; ++i) {
        // cp += S[d,t,i] * _memory_costs[i];
        const string name_rhs_cp =
            get_fixed_length_var_name("S" + idx_to_string(d, t, i));
        add_marker("B", name_rhs_cp, c_initMemory, -_memory_costs[i],
                   vector<size_t>{d, t, i});
      }
      // lhs += U[d,t,0]
      const string name_lhs_U =
          get_fixed_length_var_name("U" + idx_to_string(d, t, 0));
      add_marker("C", name_lhs_U, c_initMemory, 1, vector<size_t>{d, t, 0});

      // rhs += R[d,t,0] * _memory_costs[0] + cp;
      const string name_rhs_R =
          get_fixed_length_var_name("R" + idx_to_string(d, t, 0));
      add_marker("B", name_rhs_R, c_initMemory, -_memory_costs[0],
                 vector<size_t>{d, t, 0});
    }
  }
  // memory recurrence
  // mem_freed += _memory_costs[i] * F[(d * T * E) + t * E + e_idx];
  // lhs = U[d,t,k+1]
  // rhs = U[d,t,k] + R[d,t,k+1] * _memory_costs[k+1] - mem_freed;
  // - U[0,0,0] - 4e+06 R[0,0,1] + U[0,0,1] = 0
  //                ...                     = 0
  // model.addConstr(lhs, GRB_EQUAL, rhs);
  for (t = 0; t < T; ++t) {
    for (k = 0; k < T - 1; ++k) {
      for (d = 0; d < D; ++d) {
        add_constraint("E", 0);
        auto c_memRecurr = _constraints_insertOrder.back();
        vector<pair<size_t, size_t>> pred = _predecessors(k);
        for (auto p : pred) {
          const size_t e_idx = p.second;
          const string name_F =
              get_fixed_length_var_name("F" + idx_to_string(d, t, e_idx));
          add_marker("B", name_F, c_memRecurr, _memory_costs[i],
                     vector<size_t>{d, t, e_idx});
        }
        const string name_lhs_U =
            get_fixed_length_var_name("U" + idx_to_string(d, t, k + 1));
        const string name_rhs_U =
            get_fixed_length_var_name("U" + idx_to_string(d, t, k));
        const string name_rhs_R =
            get_fixed_length_var_name("R" + idx_to_string(d, t, k + 1));
        add_marker("C", name_lhs_U, c_memRecurr, 1,
                   vector<size_t>{d, t, k + 1});
        add_marker("C", name_rhs_U, c_memRecurr, -1, vector<size_t>{d, t, k});
        add_marker("B", name_rhs_R, c_memRecurr, -_memory_costs[k + 1],
                   vector<size_t>{d, t, k + 1});
      }
    }
  }
  // memory budget constraints
  // U[d,t,0] >= 0
  // U[d,t,0] <= gcd
  // model.addConstr(U[d,t,0], GRB_GREATER_EQUAL, 0);
  // model.addConstr(U[d,t,0], GRB_LESS_EQUAL, gcd);
  for (d = 0; d < D; ++d) {
    float gcd = static_cast<float>(_budget[d] / static_cast<float>(_ram[d]));
    for (t = 0; t < T; ++t) {
      for (i = 0; i < T; ++i) {
        add_constraint("G", 0);
        auto c_budget_lb = _constraints_insertOrder.back();
        add_constraint("L", gcd);
        auto c_budget_ub = _constraints_insertOrder.back();
        const string name_U =
            get_fixed_length_var_name("U" + idx_to_string(d, t, i));
        add_marker("C", name_U, c_budget_lb, 1, vector<size_t>{d, t, i});
        add_marker("C", name_U, c_budget_ub, 1, vector<size_t>{d, t, i});
      }
    }
  }

  for (auto c_name : _constraints_insertOrder) {
    add_row(c_name);
  }
  for (auto m : _marker) {
    add_column(m.first);
  }

  string ilp_program = "NAME " + _model_name + "\n";
  ilp_program += "ROWS\n";
  ilp_program += " N  OBJ\n" + _rows;
  ilp_program += "COLUMNS\n" + _columns;
  ilp_program += "RHS\n" + _rhs;
  ilp_program += "BOUNDS\n" + vars;
  if (_quad_obj.size() > 0) {
    string quad_obj_str = "";
    for (auto o : _quad_obj) {
      if (o.coeff > 0) {
        quad_obj_str +=
            "    " + o.var1 + "  " + o.var2 + "  " + value2str(o.coeff) + "\n";
      }
    }
    ilp_program += "QUADOBJ\n" + quad_obj_str;
  }
  ilp_program += "ENDATA\n";

  // write model to file
  if (_verbose > 0) {
    cout << "Write ilp model file " << _mpsfile << endl;
  }
  ofstream f;
  f.open(_mpsfile);
  f << ilp_program;
  f.close();
  return 0;
}

void ILP_Solver::printResults() {
  if (_verbose > 0) {
    cout << "\nSolution with computational costs: " << _objective_value
         << ", memory consumption: min " << _minimal_memory << ", peak "
         << _maximal_memory << endl;
    cout << "\nR" << endl;
    _R_matrix->print();
    cout << "\nS" << endl;
    _S_matrix->print();
  }
  if (_verbose > 1) {
    cout << "\nZ" << endl;
    _Z_matrix->print();
    cout << "\nU" << endl;
    _U_matrix->print();
    cout << "\nF" << endl;
    _F_matrix->print();
  }
}
#endif