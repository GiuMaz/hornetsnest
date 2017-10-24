/**
 * @author Samuele Germiniani e Giulio Mazzi
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         samuele.germiniani@studenti.univr.it                             <br>
 *         giulio.mazzi@studenti.univr.it
 * @date October, 2017
 * @version v1
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */

#include "Static/PageRank/PageRank.cuh"
#include "Core/Auxilary/DuplicateRemoving.cuh"
#include <GraphIO/GraphStd.hpp>
#include <GraphIO/BFS.hpp>
#include <queue>
#include <iostream>
#include "Device/WarpReduce.cuh"


namespace hornets_nest {

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct InitOperator {
    TwoLevelQueue<vid_t> queue;
    residual_t* residual;
    degree_t* out_degrees;
    rank_t* page_rank;
    rank_t initial_page_rank;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        residual[src] = 0.0;
        out_degrees[src] = vertex.degree();
        page_rank[src] = initial_page_rank;
        queue.insert(src);
    }
};

struct ResidualReset {
    residual_t* residual;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        residual[src] = 0.0;
    }
};

struct ResidualOperation {
    residual_t* residual;
    degree_t* out_degrees;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        atomicAdd(&residual[vertex.id()], (1.0f / out_degrees[dst] ));
    }
};

struct ResidualNormalization {
    residual_t* residual;
    float teleport_parameter;

    OPERATOR(Vertex& vertex) {
        residual[vertex.id()] = (1.0f - teleport_parameter ) * teleport_parameter * residual[vertex.id()];
    }
};

struct PageRankUpdate {
    rank_t* page_rank;
    residual_t* residual;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        page_rank[src] += residual[src];
    }
};

struct Normalize {
    rank_t* page_rank;
    float* norm;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        page_rank[src] /= norm[0];
    }
};

struct PageRankPropagation {
    TwoLevelQueue<vid_t> queue;
    residual_t* residual;
    degree_t* out_degrees;
    float teleport_parameter;
    float threshold;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        auto src = edge.src_id();

        auto old = residual[dst];
        atomicAdd(&residual[dst], ( teleport_parameter * residual[src] / out_degrees[src] ));

        __syncthreads();
        if ((old < threshold and residual[dst] >= threshold))
            queue.insert(dst);

    }
};

//------------------------------------------------------------------------------
/////////////
// SUPPORT //
/////////////

// from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__inline__ __device__
float blockReduceSum(float val) {

    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    xlib::WarpReduce<>::add(val);  // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) xlib::WarpReduce<>::add(val); //Final reduce within first warp

    return val;
}

__global__ void deviceReduceKernel(float *in, float* out, int N) {
    float sum = 0.0f;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < N; 
            i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        out[blockIdx.x]=sum;
}

void deviceReduce(float *in, float* out, int N) {
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceKernel<<<blocks, threads>>>(in, out, N);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

//------------------------------------------------------------------------------
//////////////
// PAGERANK //
//////////////

PageRank::PageRank(HornetGraph& hornet, HornetGraph& inverse) :
    StaticAlgorithm(hornet),
    queue(hornet),
    load_balacing(hornet),
    hornet_inverse(inverse),
    load_balacing_inverse(inverse) {
        gpu::allocate(residual, hornet.nV());
        gpu::allocate(page_rank, hornet.nV());
        gpu::allocate(out_degrees, hornet.nV());
        reset();
}

PageRank::~PageRank() {
    gpu::free(residual);
    gpu::free(page_rank);
    gpu::free(out_degrees);
    host::free(page_rank_host);
}

void PageRank::reset() {
    queue.clear();
}

void PageRank::set_parameters(float teleport, float tresh) {
    teleport_parameter = teleport;
    threshold = tresh;
}

void PageRank::run() {
    // initialize
    forAllVertices(
            hornet,
            InitOperator{
                queue,residual,out_degrees,page_rank,(1-teleport_parameter)} );

    queue.swap();

    forAllEdges(
            hornet_inverse,
            ResidualOperation{
                residual,out_degrees},
            load_balacing_inverse );

    forAllVertices(
            hornet,
            ResidualNormalization {
                residual,teleport_parameter} );

    //std::cout << "gpu residual: ";
    //gpu::printArray(residual,hornet.nV());
    //std::cout << "gpu page_rank: ";
    //gpu::printArray(page_rank,hornet.nV());
    //std::cout << "gpu out degree: ";
    //gpu::printArray(out_degrees,hornet.nV());
    //std::cout << "gpu queue: ";
    while (queue.size() > 0) {
        //queue.print();

        forAllVertices(
                hornet,
                queue,
                PageRankUpdate {
                    page_rank, residual});

        forAllEdges(
                hornet,
                queue,
                PageRankPropagation {
                queue,residual,out_degrees,teleport_parameter,threshold},
                load_balacing);

        forAllVertices(
                hornet,
                queue,
                ResidualReset {
                    residual});

        queue.swap();
    }

    float *tmp = residual;
    deviceReduce(page_rank,tmp,hornet.nV());

    //std::cout << "gpu pre normalizzazione: ";
    //gpu::printArray(page_rank,hornet.nV());

    forAllVertices(
            hornet,
            Normalize {
                page_rank,
                tmp
                });
}

void PageRank::release() {
    gpu::free(residual);
    gpu::free(page_rank);
    gpu::free(out_degrees);
    host::free(page_rank_host);
    residual = nullptr;
    page_rank = nullptr;
    out_degrees = nullptr;
    page_rank_host = nullptr;
}

void PageRank::evaluate_sequential_algorithm()
{
    host::allocate(page_rank_host,hornet.nV());
    residual_t *residual_host;
    host::allocate(residual_host,hornet.nV());

    int *out_degrees_host;
    host::allocate(out_degrees_host,hornet.nV());

    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
                                  hornet.csr_edges(), hornet.nE());

    GraphStd<vid_t, eoff_t> graph_inverse(hornet_inverse.csr_offsets(), hornet_inverse.nV(),
                                  hornet_inverse.csr_edges(), hornet_inverse.nE());

    for (size_t i = 0; i < graph.nV(); ++i)
    {
        page_rank_host[i] = 1.0f - teleport_parameter;
        residual_host[i] = 0.0f;
    }

    for (auto v : graph.V)
        out_degrees_host[v.id()] = v.out_degree();

    for (auto v : graph_inverse.V)
    {
        //std::cout << "nodo da valutare  " << v.id() << std::endl;
        for( auto e : v )
        {
            residual_host[v.id()] += 1.0f / out_degrees_host[e.dst_id()];
            //std::cout << " nodo divisore " <<  e.dst_id() << " risultato " <<
                //residual_host[v.id()] << " out degree " << out_degrees_host[e.dst_id()] << std::endl;
        }

        residual_host[v.id()] = (1.0f-teleport_parameter) * teleport_parameter * residual_host[v.id()];
    }

    std::queue<graph::GraphStd<vid_t, eoff_t>::Vertex> queue_host;

    for (auto v : graph.V)
        queue_host.push(v);

    //std::cout << "host residual: ";
    //host::printArray(residual_host,hornet.nV());
    //std::cout << "host page rank: ";
    //host::printArray(page_rank_host,hornet.nV());
    //std::cout << "host out degree: ";

    //for (auto v : graph.V)
        //std::cout << v.out_degree() << " ";
    //std::cout << std::endl<< std::endl;

    //std::cout << "host queue: ";
    //auto copy(queue_host);
    //while (!copy.empty())
    //{
        //std::cout << copy.front() << " ";
        //copy.pop();
    //}
    //std::cout << std::endl<< std::endl;

    while ( !queue_host.empty() )
    {
        auto v = queue_host.front();
        //std::cout << "\tnodo: " << v.id() << std::endl;
        queue_host.pop();
        //std::cout << "\tpage_rank vecchio: " << page_rank_host[v.id()] << " residuo " << residual_host[v.id()] << std::endl;
        page_rank_host[v.id()] += residual_host[v.id()];
        //std::cout << "\tpage_rank nuovo: " << page_rank_host[v.id()] << std::endl;
        for ( auto e : v )
        {
            //std::cout << "\t\tvicino: " << e.dst_id() << std::endl;
            residual_t old_residual_host = residual_host[e.dst_id()];
            //std::cout << "\t\told residual: " << residual_host[e.dst_id()] << std::endl;
            residual_host[e.dst_id()] +=
                ( (residual_host[v.id()] * teleport_parameter) / out_degrees_host[v.id()]);
            //std::cout << "\t\tnew residual: " << residual_host[e.dst_id()] << std::endl;

            if ( residual_host[e.dst_id()] >= threshold && 
                    old_residual_host < threshold )
            {
                //std::cout << "\t\t\taggiungi nodo " << e.dst_id() << std::endl;
                queue_host.push(e.dst());
            }
        }
        residual_host[v.id()] = 0.0f;
    }

    //std::cout << "host pre normalizzazione: ";
    //host::printArray(page_rank_host,hornet.nV());
    float norm = 0.0f;
    for (size_t i = 0; i < graph.nV(); ++i)
        norm += page_rank_host[i];

    for (size_t i = 0; i < graph.nV(); ++i)
        page_rank_host[i] /= norm;
}


bool PageRank::validate() {

    if (page_rank_host == nullptr)
        evaluate_sequential_algorithm();

    rank_t * gpu_pr;
    host::allocate(gpu_pr,hornet.nV());

    gpu::copyToHost(page_rank, hornet.nV(),gpu_pr);

    //std::cout << "valore host: ";
    //host::printArray(page_rank_host,hornet.nV());

    //std::cout << std::endl << "valore gpu: ";
    //gpu::printArray(page_rank,hornet.nV());

    //std::cout << std::endl;
    bool is_equal = true;
    for (int i = 0; i < hornet.nV(); ++i)
    {
        if ( abs(page_rank_host[i] - gpu_pr[i]) > 0.01)
        {
            is_equal = false;
            break;
        }
    }

    host::free(gpu_pr);

    return is_equal;
}

} // namespace hornets_nest
