// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file FrictionPairFrictionLJBaseGPUKernel.metal
    \brief Metal port of FrictionPairFrictionLJBaseGPUKernel.cu
*/

#include "FrictionPairGPUMetal.h"

#include "EvaluatorPairFrictionLJBase.h"
#include "EvaluatorPairFrictionLJVariants.h"

namespace hoomd
    {
namespace md
    {
#ifdef __METAL_VERSION__
HOOMD_FRICTION_PAIR_FORCE_KERNEL(gpu_compute_pair_friction_forces_kernel_linear,
                                 EvaluatorPairFrictionLJLinear)
HOOMD_FRICTION_PAIR_FORCE_KERNEL(gpu_compute_pair_friction_forces_kernel_coulomb,
                                 EvaluatorPairFrictionLJCoulomb)
HOOMD_FRICTION_PAIR_FORCE_KERNEL(gpu_compute_pair_friction_forces_kernel_coulomb_newton,
                                 EvaluatorPairFrictionLJCoulombNewton)
#else
namespace kernel
    {
template<> struct friction_kernel_name<EvaluatorPairFrictionLJLinear>
    {
    static constexpr const char* value
        = "hoomd::md::gpu_compute_pair_friction_forces_kernel_linear";
    };

template<> struct friction_kernel_name<EvaluatorPairFrictionLJCoulomb>
    {
    static constexpr const char* value
        = "hoomd::md::gpu_compute_pair_friction_forces_kernel_coulomb";
    };

template<> struct friction_kernel_name<EvaluatorPairFrictionLJCoulombNewton>
    {
    static constexpr const char* value
        = "hoomd::md::gpu_compute_pair_friction_forces_kernel_coulomb_newton";
    };

template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_friction_forces<EvaluatorPairFrictionLJLinear>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairFrictionLJLinear::param_type* d_param);

template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_friction_forces<EvaluatorPairFrictionLJCoulomb>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairFrictionLJCoulomb::param_type* d_param);

template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_friction_forces<EvaluatorPairFrictionLJCoulombNewton>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairFrictionLJCoulombNewton::param_type* d_param);
    } // end namespace kernel
#endif
    } // end namespace md
    } // end namespace hoomd
