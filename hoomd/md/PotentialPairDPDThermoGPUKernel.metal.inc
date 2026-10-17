// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal DPD force kernel for EvaluatorPairDPDThermo@_evaluator@ (generated from
// PotentialPairDPDThermoGPUKernel.metal.inc)

#include "hoomd/md/PotentialPairDPDThermoGPUMetal.h"
#include "hoomd/md/EvaluatorPairDPDThermo@_evaluator@.h"

#define EVALUATOR_CLASS EvaluatorPairDPDThermo@_evaluator@

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_DPD_FORCE_KERNEL(EVALUATOR_CLASS)
#else
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_dpd_forces<EVALUATOR_CLASS>(const dpd_pair_args_t& args,
                                        const typename EVALUATOR_CLASS::param_type* d_params);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
