// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal external force kernel for EvaluatorExternal@_evaluator@ (generated from
// PotentialExternalGPUKernel.metal.inc)

#include "hoomd/md/PotentialExternalGPUMetal.h"
#include "hoomd/md/EvaluatorExternal@_evaluator@.h"

#define EVALUATOR_CLASS EvaluatorExternal@_evaluator@

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_EXTERNAL_FORCE_KERNEL(EVALUATOR_CLASS)
#else
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_potential_external_forces<EVALUATOR_CLASS>(
    const external_potential_args_t& external_potential_args,
    const typename EVALUATOR_CLASS::param_type* d_params,
    const typename EVALUATOR_CLASS::field_type* d_field);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
