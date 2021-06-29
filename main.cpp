#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <array>
#include <chrono>
#include <nlohmann/json.hpp>
#include <icecream.hpp>
#include "wsn_with_features_positions.hpp"
#include "graph_analysis.hpp"

double toDouble(char* str) {
  double d = std::strtod(str, NULL);
  if(errno == ERANGE) {
    throw "invalid conversion";
  }
  return d;
}

long toLong(char* str) {
  long l = std::strtol(str, NULL, 0);
  if(errno == ERANGE) {
    throw "invalid conversion";
  }
  return l;
}

int level = 0;

template <class F>
void MeasureElapsed(std::string tag, F f) {
  level++;
  auto start = std::chrono::system_clock::now();
  f();
  auto end = std::chrono::system_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  for(int i=0; i<level-1; i++) { std::cerr << "  "; } // indent
  std::cerr << "elapsed time for " << tag << ": " << dt/1000.0 << std::endl;
  level--;
}

struct param_t {
  public:
  long net_size;
  double p_tri;
  double p_r;
  double p_nd;
  double p_ld;
  double aging;
  double w_th;
  double w_r;
  size_t q;
  size_t F;
  double alpha;
  long t_max;
  long _seed;
  void Print(std::ostream& os) const {
    os << net_size << ' ' << p_tri << ' ' << p_r << ' ' << p_nd << ' '
       << p_ld << ' ' << aging << ' ' << w_th << ' ' << w_r << ' '
       << q << ' ' << F << ' ' << alpha << ' ' << t_max << ' ' << _seed << ' ';
  }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(param_t, net_size, p_tri, p_r, p_nd, p_ld, aging, w_th, w_r, q, F, alpha, t_max, _seed);

struct output_t {
  public:
  double average_degree, stddev_degree, average_link_weight, pcc_k_knn, clustering_coefficient, pcc_c_k,
         link_overlap, pcc_link_overlap_weight, percolation_fc_ascending, percolation_fc_descending;
  output_t() : average_degree(-1.0), stddev_degree(-1.0), average_link_weight(-1.0), pcc_k_knn(-1.0),
               clustering_coefficient(-1.0), pcc_c_k(-1.0), link_overlap(-1.0), pcc_link_overlap_weight(-1.0),
               percolation_fc_ascending(-1.0), percolation_fc_descending(-1.0) {};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(output_t, average_degree, stddev_degree, average_link_weight, pcc_k_knn, clustering_coefficient,
                                   pcc_c_k, link_overlap, pcc_link_overlap_weight, percolation_fc_ascending, percolation_fc_descending);

int main( int argc, char** argv) {
  param_t p;
  if( argc == 2) {
    std::ifstream fin(argv[1]);
    std::cerr << argv[1] << std::endl;
    if (!fin) {
      std::cerr << "Failed to open " << argv[1] << std::endl;
      throw std::runtime_error("invalid input file");
    }
    nlohmann::json j;
    fin >> j;
    from_json(j, p);
  }
  else {
    std::cerr << "Usage : ./wsn.out <input.json>" << std::endl;
    std::cerr << argc << std::endl;
    exit(1);
  }
  {
    nlohmann::json j;
    to_json(j, p);
    std::cerr << "Lists of given parameters are as follows:" << std::endl << j.dump(2);
  }

  typedef NodeWithFeaturesPositions N_T;
  WSNWithFeaturesPositions<N_T> sim(p._seed, p.net_size, p.p_tri, p.p_r, p.w_r, p.p_nd, p.p_ld, p.aging, p.w_th, p.q, p.F, p._seed+1234, p.alpha, p._seed+2345);

  const double cutoff_k = 150;
  bool ret = false;
  MeasureElapsed("simulation", [&] {
    ret = sim.Run(p.t_max, cutoff_k);
  });

  if (!ret) {
    throw std::runtime_error("Simulation failed");
  }

  auto g = sim.GetGraph();

  std::ofstream fout("_output.json");
  output_t output;
  MeasureElapsed("observation", [&] {
    output.average_degree = g->AverageDegree();
    output.stddev_degree  = g->StddevDegree();
    output.average_link_weight = g->AverageStrength() / g->AverageDegree();
    output.pcc_k_knn = g->PCC_k_knn();
    auto cc_pccck = g->CC_and_PCC_Ck();
    output.clustering_coefficient = cc_pccck.first;
    output.pcc_c_k = cc_pccck.second;
    auto o_pccow = g->O_and_PCC_Ow();
    output.link_overlap = o_pccow.first;
    output.pcc_link_overlap_weight = o_pccow.second;

    GraphAnalysis<NodeWithFeaturesPositions> a(*g);
    auto results = a.AnalyzeLinkRemovalPercolationVariableAccuracy(0.01, 0.02);
    output.percolation_fc_ascending = (1.0 - results.first.Fc()) * output.average_degree;
    output.percolation_fc_descending = (1.0 - results.second.Fc()) * output.average_degree;
  });

  nlohmann::json j_output;
  to_json(j_output, output);
  fout << j_output.dump(2) << std::endl;
  fout.flush();
  fout.close();

  return 0;
}

