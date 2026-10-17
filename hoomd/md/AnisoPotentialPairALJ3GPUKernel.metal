// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file AnisoPotentialPairALJ3GPUKernel.metal
    \brief Metal port of AnisoPotentialPairALJ3GPUKernel.cu
*/

#include "AnisoPotentialPairGPUMetal.h"
#include "EvaluatorPairALJ.h"

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_ANISO_PAIR_FORCE_KERNELS(EvaluatorPairALJ<3>)
#else
namespace kernel
    {
template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairALJ<3>>(const a_pair_args_t& pair_args,
                                      const EvaluatorPairALJ<3>::param_type* d_param,
                                      const EvaluatorPairALJ<3>::shape_type* d_shape_param);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
