/**
 * @brief PageRank algorithm implementation based on a push-driven mechanics
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
 *
 * @file
 */
#pragma once

#include "HornetAlg.hpp"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

namespace hornets_nest {

using HornetGraph = gpu::Csr<EMPTY, EMPTY>;

using residual_t = float;
using rank_t = float;
using dist_t = int;

class PageRank : public StaticAlgorithm<HornetGraph> {
public:
    PageRank(HornetGraph& hornet, HornetGraph& inverse);
    ~PageRank();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;
    void set_parameters(float teleport, float tresh);
    void evaluate_sequential_algorithm();


private:
    TwoLevelQueue<vid_t>        queue;
    load_balancing::BinarySearch load_balacing;
    //load_balancing::VertexBased1 load_balacing;
    //load_balancing::BinarySearch load_balacing_inverse;
    load_balancing::VertexBased1 load_balacing_inverse;
    //load_balancing::ScanBased load_balacing_inverse;

    residual_t* actual_residual   { nullptr };
    residual_t* new_residual      { nullptr };
    rank_t* page_rank             { nullptr };
    degree_t* out_degrees         { nullptr };
    rank_t* page_rank_host        { nullptr };
    float teleport_parameter      {0.85};
    float threshold               {0.01};
    HornetGraph& hornet_inverse;
};

} // namespace hornets_nest
