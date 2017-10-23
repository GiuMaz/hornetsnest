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
        residual[dst] += 1.0 + (1.0 / out_degrees[dst] );
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
    float norm;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        page_rank[src]/=norm;
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
        residual[dst] += 
            ( teleport_parameter * residual[src] / out_degrees[src] );
        if (old < threshold and residual[dst] >= threshold)
            queue.insert(dst);
    }
};

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

    while (queue.size() > 0) {
        queue.print();

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

    float norm = 1.0; //gpu::reduce(page_rank,hornet.nV());
    forAllVertices(
            hornet,
            Normalize {
                page_rank,
                norm});

}

void PageRank::release() {
    gpu::free(residual);
    gpu::free(page_rank);
    gpu::free(out_degrees);
    residual = nullptr;
    page_rank = nullptr;
    out_degrees = nullptr;
}

bool PageRank::validate() {

    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
                                  hornet.csr_edges(), hornet.nE());

    GraphStd<vid_t, eoff_t> graph_inverse(hornet_inverse.csr_offsets(), hornet_inverse.nV(),
                                  hornet_inverse.csr_edges(), hornet_inverse.nE());

    float *page_rank_host = new float(graph.nV());
    float *residual_host = new float(graph.nV());

    for (size_t i = 0; i < graph.nV(); ++i)
    {
        page_rank_host[i] = 1.0f - teleport_parameter;
        residual_host[i] = 0.0f;
    }

    for (auto v : graph_inverse.V)
    {
        for( auto e : v )
            residual_host[v.id()] += 1.0f / v.out_degree();

        residual_host[v.id()] += 1.0f / v.out_degree();
    }

    std::queue<GraphStd<vid_t, eoff_t>::Vertex> queue_host;
    for (auto v : graph_inverse.V)
        queue_host.push(v);

    while ( !queue_host.empty() )
    {
        auto v = queue_host.front();
        queue_host.pop();
        auto new_page_rank_v = page_rank_host[v.id()] + residual_host[v.id()];
        for ( auto e : v )
        {
            auto old_residual_host = residual_host[e.dst_id()];
            residual_host[e.dst_id()] = residual_host[e.dst_id()] +
                ( (residual_host[v.id()] * teleport_parameter) / v.out_degree());

            if ( residual_host[e.dst_id()] >= threshold and 
                    old_residual_host < threshold )
                queue_host.push(e.dst());
        }
        residual_host[v.id()] = 0.0f;
    }

    float norm = 0.0f;
    for (size_t i = 0; i < graph.nV(); ++i)
        norm += page_rank_host[i];

    for (size_t i = 0; i < graph.nV(); ++i)
        page_rank_host[i] += page_rank_host[i] / norm;

    return false;
}

} // namespace hornets_nest
