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

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);
    //graph.print();

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.out_offsets_ptr(),
                           graph.out_edges_ptr());

    HornetGraph hornet_graph(hornet_init);
    //hornet_graph.print();

    PageRank page_rank(hornet_graph);

    page_rank.set_parameters(0.85);

    Timer<DEVICE> TM;
    cudaProfilerStart();
    TM.start();

    page_rank.run();

    TM.stop();
    cudaProfilerStop();
    TM.print("TopDown");

    auto is_correct = page_rank.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");

    return !is_correct;
}
