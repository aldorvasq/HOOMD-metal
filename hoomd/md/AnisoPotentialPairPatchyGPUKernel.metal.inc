// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal patchy pair kernel for EvaluatorPair@_evaluator@ (generated from
// AnisoPotentialPairPatchyGPUKernel.metal.inc)

#include "hoomd/md/AnisoPotentialPairGPUMetal.h"
#include "hoomd/md/PairModulator.h"
#include "hoomd/md/PatchEnvelope.h"

#include "hoomd/md/EvaluatorPair@_evaluator@.h"

#define EVALUATOR_CLASS EvaluatorPair@_evaluator@

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_ANISO_PAIR_FORCE_KERNELS(PairModulator<EVALUATOR_CLASS, PatchEnvelope>)
#else
namespace kernel
    {
template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<PairModulator<EVALUATOR_CLASS, PatchEnvelope>>(
    const a_pair_args_t& pair_args,
    const PairModulator<EVALUATOR_CLASS, PatchEnvelope>::param_type* d_param,
    const PairModulator<EVALUATOR_CLASS, PatchEnvelope>::shape_type* d_shape_param);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
