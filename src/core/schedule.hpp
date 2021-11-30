#ifndef XENGINE_SCHEDULE_HPP
#define XENGINE_SCHEDULE_HPP

using namespace std;
using namespace dnnl;

class MemoryLayoutTag {
public:
  MemoryLayoutTag() { dnnl_memory_tag = ft::any; }
  MemoryLayoutTag(const string s) {
    unordered_map<string, ft> map = {{pair<string, ft>("any", ft::any)},
                                     {pair<string, ft>("nc", ft::nc)},
                                     {pair<string, ft>("nchw", ft::nchw)},
                                     {pair<string, ft>("nhwc", ft::nhwc)},
                                     {pair<string, ft>("ncdhw", ft::ncdhw)},
                                     {pair<string, ft>("ndhwc", ft::ndhwc)}};
    dnnl_memory_tag = map[s];
  }
  ~MemoryLayoutTag() {}

  ft to_dnnl() { return dnnl_memory_tag; }

  string to_string() {
    unordered_map<ft, string> map = {{pair<ft, string>(ft::any, "any")},
                                     {pair<ft, string>(ft::nc, "nc")},
                                     {pair<ft, string>(ft::nchw, "nchw")},
                                     {pair<ft, string>(ft::nhwc, "nhwc")},
                                     {pair<ft, string>(ft::ncdhw, "ncdhw")},
                                     {pair<ft, string>(ft::ndhwc, "ndhwc")}};
    return map[dnnl_memory_tag];
  }

  ft dnnl_memory_tag;
};

enum DecisionType {
  PRODUCE_TENSOR,
  SAVE_TENSOR,
  FREE_TENSOR,
  TRANSFORM_LAYOUT
};

class ScheduleDecision {
public:
  ScheduleDecision() { tensorID = "unknown"; }
  ScheduleDecision(string tID) : tensorID(tID) {}
  virtual ~ScheduleDecision(){};
  virtual void print() = 0;
  DecisionType type;
  string tensorID;
};

class ExecuteOperator : public ScheduleDecision {
public:
  ExecuteOperator(string opID, string eID, int sID, const string tag)
      : operatorID(opID), engineID(eID), streamID(sID) {
    type = DecisionType::PRODUCE_TENSOR;
    outputTag = MemoryLayoutTag(tag);
  }
  ~ExecuteOperator() {}
  void print() {
    cout << operatorID << ";" << engineID << ";" << streamID << ";"
         << outputTag.to_string() << endl;
  }
  string operatorID;
  string engineID;
  int streamID;
  MemoryLayoutTag outputTag;
};

class Schedule {
public:
  Schedule() { decisions = vector<unique_ptr<ScheduleDecision>>(); }
  Schedule(const string &schedule_path) {
    decisions = vector<unique_ptr<ScheduleDecision>>();
    vector<vector<string>> info = parse_input_file(schedule_path);
    for (auto i : info) {
      const string opID = i.at(0);
      const string engID = i.at(1);
      const int streamID = stoi(i.at(2));
      const string tag = i.at(3);
      decisions.push_back(
          move(make_unique<ExecuteOperator>(opID, engID, streamID, tag)));
    }
  };
  size_t size() { return decisions.size(); }
  ScheduleDecision *get(const int opID) { return decisions.at(opID).get(); }
  ~Schedule() {
    for (auto i = 0; i < decisions.size(); i++) {
      decisions.at(i).reset();
      decisions.at(i).release();
    }
  }
  void print() {
    for (auto i = 0; i < decisions.size(); i++) {
      if (decisions.at(i)->type == DecisionType::PRODUCE_TENSOR) {
        decisions.at(i)->print();
      }
    }
  }

private:
  vector<unique_ptr<ScheduleDecision>> decisions;
};
#endif