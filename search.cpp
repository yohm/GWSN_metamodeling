#include <iostream>
#include <cstdlib>
#include <array>
#include <chrono>
#include <caravan.hpp>
#include <icecream/icecream.hpp>
#include "wsn_with_features_positions.hpp"
#include "graph_analysis.hpp"

int g_level = 0;

template <class F>
void MeasureElapsed(std::string tag, F f) {
  g_level++;
  auto start = std::chrono::system_clock::now();
  f();
  auto end = std::chrono::system_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  for(int i=0; i<g_level-1; i++) { std::cerr << "  "; } // indent
  std::cerr << "elapsed time for " << tag << ": " << dt/1000.0 << std::endl;
  g_level--;
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
  uint64_t seed;
  void Print(std::ostream& os) const {
    os << net_size << ' ' << p_tri << ' ' << p_r << ' ' << p_nd << ' '
       << p_ld << ' ' << aging << ' ' << w_th << ' ' << w_r << ' '
       << q << ' ' << F << ' ' << alpha << ' ' << t_max << ' ' << seed << ' ';
  }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(param_t, net_size, p_tri, p_r, p_nd, p_ld, aging, w_th, w_r, q, F, alpha, t_max, seed);

struct output_t {
  public:
  double degree_average;
  double degree_stddev;
  double average_link_weight;
  double pcc_k_knn;
  double clustering_coefficient;
  double pcc_clustering_k;
  double link_overlap;
  double pcc_overlap_weight;
  double percolation_fc_ascending;
  double percolation_fc_descending;
  output_t() {
    degree_average = -1.0;
    degree_stddev = -1.0;
    average_link_weight = -1.0;
    pcc_k_knn = -1.0;
    clustering_coefficient = -1.0;
    pcc_clustering_k = -1.0;
    link_overlap = -1.0;
    pcc_overlap_weight = -1.0;
    percolation_fc_ascending = -1.0;
    percolation_fc_descending = -1.0;
  }
  void Print(std::ostream& os) const {
    os << degree_average << ' ' << degree_stddev << ' ' << average_link_weight << ' ' << pcc_k_knn << ' '
       << clustering_coefficient << ' ' << pcc_clustering_k << ' ' << link_overlap << ' '
       << pcc_overlap_weight << ' ' << percolation_fc_ascending << ' ' << percolation_fc_descending << ' ';
  }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(output_t, degree_average, degree_stddev, average_link_weight, pcc_k_knn, clustering_coefficient, pcc_clustering_k, link_overlap, pcc_overlap_weight, percolation_fc_ascending, percolation_fc_descending);

using io_vec_t = std::vector<std::pair<param_t, output_t>>;

io_vec_t ExecuteSimulation(const param_t& p) {
  typedef NodeWithFeaturesPositions N_T;
  WSNWithFeaturesPositions<N_T> sim(p.seed, p.net_size, p.p_tri, p.p_r, p.w_r, p.p_nd, p.p_ld, p.aging, p.w_th, p.q, p.F, p.seed+1234, p.alpha, p.seed+2345);

  const double cutoff_k = 150;
  const long dt = p.t_max / 10;
  io_vec_t ans_vec;

  MeasureElapsed("simulation", [&] {
    for (long t = 0; t < p.t_max; t+= dt) {
      bool success = sim.Run(dt, cutoff_k);
      if (!success) { break; }  // when k > cutoff_k, break

      auto g = sim.GetGraph();
      output_t output;
      MeasureElapsed("observation", [&] {
        output.degree_average      = g->AverageDegree();
        output.degree_stddev       = g->StddevDegree();
        output.average_link_weight = g->AverageStrength() / g->AverageDegree();
        output.pcc_k_knn           = g->PCC_k_knn();
        auto cc_pccck = g->CC_and_PCC_Ck();
        output.clustering_coefficient = cc_pccck.first;
        output.pcc_clustering_k       = cc_pccck.second;
        auto o_pccow = g->O_and_PCC_Ow();
        output.link_overlap       = o_pccow.first;
        output.pcc_overlap_weight = o_pccow.second;

        GraphAnalysis<NodeWithFeaturesPositions> a(*g);
        auto results = a.AnalyzeLinkRemovalPercolationVariableAccuracy(0.01, 0.02);
        output.percolation_fc_ascending = (1.0 - results.first.Fc()) * output.degree_average;
        output.percolation_fc_descending = (1.0 - results.second.Fc()) * output.degree_average;
      });
      param_t in = p;
      in.t_max = t + dt;
      ans_vec.emplace_back( std::make_pair(in, output) );
    }

  });

  return ans_vec;
}

std::vector<nlohmann::json> GenerateParam(std::mt19937_64& rnd, long N_sample) {
  param_t param;
  param.net_size = std::uniform_int_distribution<size_t>(1000,5000)(rnd);
  param.p_tri = std::pow(10.0, std::uniform_real_distribution<double>(-3.0, 0.0)(rnd));
  param.p_r = std::pow(10.0, std::uniform_real_distribution<double>(-4.0, -2.0)(rnd));
  param.p_nd = std::pow(10.0, std::uniform_real_distribution<double>(-4.0, -2.0)(rnd));
  param.p_ld = std::pow(10.0, std::uniform_real_distribution<double>(-4.0, -2.0)(rnd));
  param.aging = std::pow(10.0, std::uniform_real_distribution<double>(-4.0, -1.0)(rnd));
  param.w_th = 0.5;
  param.w_r = std::uniform_real_distribution<double>(0.0, 2.0)(rnd);
  param.q = std::uniform_int_distribution<int>(1, 10)(rnd);
  param.F = std::uniform_int_distribution<int>(1, 10)(rnd);
  param.alpha = std::uniform_real_distribution<double>(0.0, 4.0)(rnd);
  param.t_max = 50000;

  std::vector<nlohmann::json> ret;
  for (long j = 0; j < N_sample; j++) {
    param.seed = rnd();
    ret.emplace_back(param);
  }
  return ret;
}

int main( int argc, char** argv) {
  MPI_Init(&argc, &argv);

  using namespace nlohmann;

  if (argc != 5) {
    std::cerr << "invalid number of arguments: " << argc << "\n";
    std::cerr << "Usage : " << argv[0] << "<N_init> <duration(sec)> <N_sample> <seed>" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int my_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int num_procs = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  long N_init = std::stol(argv[1]);
  long duration = std::stol(argv[2]);
  long N_sample = std::stol(argv[3]);
  uint64_t seed = std::stoull(argv[4]);

  if (my_rank == 0) {
    std::cerr << "Lists of given parameters are as follows:" << std::endl;
    IC(N_init, duration, N_sample, seed);
  }

  std::mt19937_64 rnd(seed);
  auto start_time = std::chrono::system_clock::now();
  auto timeout_time = start_time + std::chrono::seconds(duration);

  std::function<void(caravan::Queue&)> on_init = [N_init,N_sample,&rnd](caravan::Queue& q) {
    // randomly set input parameters from the specified range
    for (long i = 0; i < N_init; i++) {
      std::vector<json> params = GenerateParam(rnd, N_sample);
      for (const auto& param: params) {
        q.Push(param);
      }
    }
  };

  io_vec_t IO_vec;

  std::function<void(int64_t, const json&, const json&, caravan::Queue&)> on_result_receive = [&IO_vec,&rnd,N_sample,timeout_time](int64_t task_id, const json& input, const json& output, caravan::Queue& q) {
    for (const auto& io: output) {
      param_t in;
      output_t out;
      ::from_json(io[0], in);
      ::from_json(io[1], out);
      IO_vec.emplace_back(std::make_pair(in,out));
    }
    if (IO_vec.size() >= 100 || q.Size() == 0) {
      for (const auto& x: IO_vec) {
        x.first.Print(std::cout);
        x.second.Print(std::cout);
        std::cout << "\n";
      }
      std::cout << std::flush;
      IO_vec.clear();
    }
    auto now = std::chrono::system_clock::now();
    if (now < timeout_time && q.Size() == 0) {
      std::cerr << "additional job" << std::endl;
      std::cerr << std::chrono::duration_cast<std::chrono::seconds>(timeout_time - now).count() << " sec until timeout" << std::endl;
      std::vector<json> params = GenerateParam(rnd, N_sample);
      for (const auto& param: params) { q.Push(param); }
    }
  };

  std::function<json(const json& input)> do_task = [](const json& input) {
    param_t param;
    ::from_json(input, param);
    io_vec_t io_vec = ExecuteSimulation(param);
    json output_json(io_vec);
    return output_json;
  };

  caravan::Option opt;
  // opt.dump_log = "tasks.msgpack";
  caravan::Start(on_init, on_result_receive, do_task, MPI_COMM_WORLD, opt);

  MPI_Finalize();

  return 0;
}

