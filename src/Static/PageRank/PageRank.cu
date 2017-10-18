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

namespace hornets_nest {

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct InitOperator {
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex) {
        vid_t src = vertex.id();
        queue.insert(src);
    }
};

struct PageRankOperator {
    OPERATOR(Vertex& vertex, Edge& edge) {
    }
};
//------------------------------------------------------------------------------
/////////////////
// PageRank//
/////////////////

PageRank::PageRank(HornetGraph& hornet) :
    StaticAlgorithm(hornet),
    queue(hornet),
    load_balacing(hornet) {

        gpu::allocate(page_rank, hornet.nV());
        gpu::allocate(residual, hornet.nV());
        reset();
}

PageRank::~PageRank() {

}

void PageRank::reset() {

}

void PageRank::set_parameters(float teleport) {
    teleport_parameter = teleport;
    forAllVertices(hornet, InitOperator{queue} );
    queue.swap();
    queue.print();
}

void PageRank::run() {
    // initialize
    // TODO

    queue.print();
    while (queue.size() > 0) {
        //std::cout << queue.size() << std::endl;
        //for all edges in "queue" applies the operator "BFSOperator" by using
        //the load balancing algorithm instantiated in "load_balacing"
        forAllEdges(hornet, queue,
                    PageRankOperator {},
                    load_balacing);
        queue.swap();
    }

}

void PageRank::release() {

}

bool PageRank::validate() {
    return false;
}

} // namespace hornets_nest
