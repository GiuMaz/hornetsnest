/**
 * @brief PageRank test program
 * @file
 */
#include "Static/PageRank/PageRank.cuh"
#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;
    using namespace graph;
    using namespace graph::structure_prop;

    GraphStd<vid_t, eoff_t> graph(ENABLE_INGOING);
    CommandLineParam cmd(graph, argc, argv);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);

    HornetInit hornet_init_inverse(graph.nV(), graph.nE(),
                                   graph.csr_in_offsets(),
                                   graph.csr_in_edges());

    HornetGraph hornet_graph_inverse(hornet_init_inverse);

    PageRank page_rank(hornet_graph, hornet_graph_inverse);

    page_rank.set_parameters(0.85, 0.001);

    Timer<DEVICE> TM;
    cudaProfilerStart();
    TM.start();

    page_rank.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("device PageRank");

    Timer<HOST> TM2;
    cudaProfilerStart();
    TM2.start();

    page_rank.evaluate_sequential_algorithm();

    TM2.stop();
    cudaProfilerStop();
    TM2.print("host PageRank");


    std::cout << "speedup: " << (TM2.duration()/TM.duration()) << "x\n";
    auto is_correct = page_rank.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");

    return 0;
}

