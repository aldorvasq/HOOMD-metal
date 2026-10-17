// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal pair force kernel for EvaluatorPair@_evaluator_cpp@ (generated from
// PotentialPairGPUKernel.metal.inc)

#include "hoomd/md/PotentialPairGPUMetal.h"
#include "hoomd/md/EvaluatorPair@_evaluator_cpp@.h"

#define EVALUATOR_CLASS EvaluatorPair@_evaluator_cpp@

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_PAIR_FORCE_KERNELS(EVALUATOR_CLASS)
#else
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_pair_forces<EVALUATOR_CLASS>(const pair_args_t& pair_args,
                                         const EVALUATOR_CLASS::param_type* d_params);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
