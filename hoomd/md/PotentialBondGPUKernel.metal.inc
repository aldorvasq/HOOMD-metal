// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Metal bond force kernel for EvaluatorBond@_bond@ (generated from
// PotentialBondGPUKernel.metal.inc)

#include "hoomd/md/PotentialBondGPUMetal.h"
#include "hoomd/md/EvaluatorBond@_bond@.h"

#define EVALUATOR_CLASS EvaluatorBond@_bond@

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_BOND_FORCE_KERNEL(EVALUATOR_CLASS, 2)
#else
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_bond_forces<EVALUATOR_CLASS, 2>(const kernel::bond_args_t<2>& bond_args,
                                            const typename EVALUATOR_CLASS::param_type* d_params,
                                            unsigned int* d_flags);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
