// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file AnisoPotentialPairGBGPUKernel.metal
    \brief Metal port of AnisoPotentialPairGBGPUKernel.cu
*/

#include "AnisoPotentialPairGPUMetal.h"
#include "EvaluatorPairGB.h"

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_ANISO_PAIR_FORCE_KERNELS(EvaluatorPairGB)
#else
namespace kernel
    {
template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairGB>(const a_pair_args_t& pair_args,
                                      const EvaluatorPairGB::param_type* d_param,
                                      const EvaluatorPairGB::shape_type* d_shape_param);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
