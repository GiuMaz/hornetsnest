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
#include "Device/Primitives/WarpReduce.cuh"


namespace hornets_nest {

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct InitOperator {
    TwoLevelQueue<vid_t> queue;
    residual_t* actual_residual;
    residual_t* new_residual;
    degree_t* out_degrees;
    rank_t* page_rank;
    rank_t initial_page_rank;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        actual_residual[src] = 0.0f;
        new_residual[src] = 0.0f;
        out_degrees[src] = vertex.degree();
        page_rank[src] = initial_page_rank;
        queue.insert(src);
    }
};

struct ResidualReset {
    residual_t* residual;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        residual[src] = 0.0f;
    }
};

struct MoveResidual {
    residual_t* actual_residual;
    residual_t* new_residual;
    TwoLevelQueue<vid_t> queue;
    float threshold;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        actual_residual[src] += new_residual[src];
        new_residual[src] = 0.0f;

        if (actual_residual[vertex.id()] >= threshold)
            queue.insert(vertex.id());
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
        residual[vertex.id()] = (1.0f - teleport_parameter ) *
            teleport_parameter * residual[vertex.id()];
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
    residual_t* actual_residual;
    residual_t* new_residual;
    degree_t* out_degrees;
    float teleport_parameter;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        auto src = edge.src_id();

        atomicAdd(&new_residual[dst], (teleport_parameter *
                    (actual_residual[src] / out_degrees[src])));
    }
};

//------------------------------------------------------------------------------
/////////////
// SUPPORT //
/////////////

// from
// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
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
        gpu::allocate(actual_residual, hornet.nV());
        gpu::allocate(new_residual, hornet.nV());
        gpu::allocate(page_rank, hornet.nV());
        gpu::allocate(out_degrees, hornet.nV());
        reset();
}

PageRank::~PageRank() {
    gpu::free(actual_residual);
    gpu::free(new_residual);
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

    forAllVertices(
            hornet,
            InitOperator{
                queue,actual_residual, new_residual,out_degrees,
                page_rank,(1-teleport_parameter)} );

    queue.swap();

    forAllEdges(
            hornet_inverse,
            ResidualOperation{ actual_residual,out_degrees},
            load_balacing_inverse );

    forAllVertices(
            hornet,
            ResidualNormalization { actual_residual,teleport_parameter} );

    while (queue.size() > 0) {

        forAllVertices(
                hornet,
                queue,
                PageRankUpdate { page_rank, actual_residual});

        forAllEdges(
                hornet,
                queue,
                PageRankPropagation {
                    actual_residual,new_residual,out_degrees,teleport_parameter},
                load_balacing);

        forAllVertices(
                hornet,
                queue,
                ResidualReset { actual_residual });

        forAllVertices(
                hornet,
                MoveResidual {
                    actual_residual, new_residual, queue, threshold });

        queue.swap();
    }

    float *tmp = actual_residual;
    deviceReduce(page_rank,tmp,hornet.nV());

    forAllVertices(
            hornet,
            Normalize {
                page_rank,
                tmp
                });
}

void PageRank::release() {
    gpu::free(actual_residual);
    gpu::free(new_residual);
    gpu::free(page_rank);
    gpu::free(out_degrees);
    host::free(page_rank_host);
    actual_residual = nullptr;
    new_residual = nullptr;
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

    GraphStd<vid_t, eoff_t> graph_inverse(hornet_inverse.csr_offsets(),
            hornet_inverse.nV(), hornet_inverse.csr_edges(), hornet_inverse.nE());

    for (size_t i = 0; i < graph.nV(); ++i)
    {
        page_rank_host[i] = 1.0f - teleport_parameter;
        residual_host[i] = 0.0f;
    }

    for (auto v : graph.V)
        out_degrees_host[v.id()] = v.out_degree();

    for (auto v : graph_inverse.V)
    {
        for( auto e : v )
        {
            residual_host[v.id()] += 1.0f / out_degrees_host[e.dst_id()];
        }

        residual_host[v.id()] = (1.0f-teleport_parameter) *
            teleport_parameter * residual_host[v.id()];
    }

    std::queue<graph::GraphStd<vid_t, eoff_t>::Vertex> queue_host;

    for (auto v : graph.V)
        queue_host.push(v);

    while ( !queue_host.empty() )
    {
        auto v = queue_host.front();
        queue_host.pop();
        page_rank_host[v.id()] += residual_host[v.id()];
        for ( auto e : v )
        {
            residual_t old_residual_host = residual_host[e.dst_id()];
            residual_host[e.dst_id()] +=
                ( (residual_host[v.id()] * teleport_parameter) /
                  out_degrees_host[v.id()]);

            if ( (residual_host[e.dst_id()] >= threshold) && 
                    (old_residual_host < threshold) )
            {
                queue_host.push(e.dst());
            }
        }
        residual_host[v.id()] = 0.0f;
    }

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

    std::cout << "host values (first 20): ";
    host::printArray(page_rank_host,min(20,hornet.nV()));

    std::cout << std::endl << "device values (first 20): ";
    gpu::printArray(page_rank,min(20,hornet.nV()));

    std::cout << std::endl;
    bool is_equal = true;
    int number_of_error = 0;
    int errori_host_maggiore = 0;
    int errori_device_maggiore  = 0;

    float tot_host = 0.0f;
    for (int i = 0; i < hornet.nV(); ++i)
        tot_host += page_rank_host[i];

    float tot_device = 0.0f;
    for (int i = 0; i < hornet.nV(); ++i)
        tot_device += gpu_pr[i];

    std::cout << "totale host: " << tot_host << " totale device: "
        << tot_device << std::endl;

    for (int i = 0; i < hornet.nV(); ++i)
    {
        if ( abs(page_rank_host[i] - gpu_pr[i])/page_rank_host[i] > 0.3 )
        {
            ++number_of_error;
            if (gpu_pr[i] > page_rank_host[i])
                ++errori_device_maggiore;
            else
                ++errori_host_maggiore;

            is_equal = false;
        }
    }

    if (number_of_error > 0)
        std::cout << "errors percentage: " << (number_of_error * 100.0) /
            hornet.nV()<<"%" << std::endl;

    host::free(gpu_pr);

    return is_equal;
}

} // namespace hornets_nest
