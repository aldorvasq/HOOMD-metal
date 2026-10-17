// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_FRICTIONLJVARIANTS_H__
#define __PAIR_EVALUATOR_FRICTIONLJVARIANTS_H__

/*! \file EvaluatorPairFrictionLJVariants.h
    \brief Defines the different variants of frictional contact interactions
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc.  HOSTDEVICE is __host__ __device__ when included in nvcc and blank
// when included into the host compiler

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {

//! Friction law of EvaluatorPairFrictionLJLinear
struct FrictionLJLinearLaw
    {
    HOSTDEVICE static void eval_factors(Scalar HOOMD_THREAD& factor_f,
                                        Scalar HOOMD_THREAD& factor_r,
                                        Scalar w,
                                        Scalar du,
                                        Scalar gamma,
                                        Scalar kappa,
                                        Scalar pair_temp,
                                        Scalar nu_ij)
        {
        Scalar f = gamma;

        factor_f = w * f;
        factor_r = fast::sqrt(factor_f);
        }

#if !defined(__HIPCC__) && !defined(__METAL_VERSION__)
    static std::string getName()
        {
        return "FrictionLJLinear";
        }
#endif
    };

//! Friction law of EvaluatorPairFrictionLJCoulomb
struct FrictionLJCoulombLaw
    {
    HOSTDEVICE static void eval_factors(Scalar HOOMD_THREAD& factor_f,
                                        Scalar HOOMD_THREAD& factor_r,
                                        Scalar w,
                                        Scalar du,
                                        Scalar gamma,
                                        Scalar kappa,
                                        Scalar pair_temp,
                                        Scalar nu_ij)
        {
        Scalar f = kappa / du;
        Scalar D = Scalar(0.0);
        if (pair_temp > 0.0)
            D = kappa * fast::sqrt(M_PI / (Scalar(2.0) * pair_temp * nu_ij))
                * exp(du * du / (2 * pair_temp * nu_ij))
                * erfc(du / fast::sqrt(Scalar(2.0) * pair_temp * nu_ij));

        factor_f = w * f;
        factor_r = fast::sqrt(w * D);
        }

#if !defined(__HIPCC__) && !defined(__METAL_VERSION__)
    static std::string getName()
        {
        return "FrictionLJCoulomb";
        }
#endif
    };

//! Friction law of EvaluatorPairFrictionLJCoulombNewton
struct FrictionLJCoulombNewtonLaw
    {
    HOSTDEVICE static void eval_factors(Scalar HOOMD_THREAD& factor_f,
                                        Scalar HOOMD_THREAD& factor_r,
                                        Scalar w,
                                        Scalar du,
                                        Scalar gamma,
                                        Scalar kappa,
                                        Scalar pair_temp,
                                        Scalar nu_ij)
        {
        Scalar f = Scalar(0.0);
        Scalar D = Scalar(0.0);

        if (du < (w * kappa / gamma))
            {
            if (pair_temp > 0.0)
                D = w * kappa * fast::sqrt(M_PI / (Scalar(2.0) * pair_temp * nu_ij))
                        * exp(du * du / (Scalar(2.0) * pair_temp * nu_ij))
                        * erfc(w * kappa / (gamma * fast::sqrt(Scalar(2.0) * pair_temp * nu_ij)))
                    + gamma
                          * (Scalar(1.0)
                             - exp((du * du - (w * kappa / gamma) * (w * kappa / gamma))
                                   / (Scalar(2.0) * pair_temp * nu_ij)));
            f = gamma;
            }
        else
            {
            if (pair_temp > 0.0)
                D = w * kappa * fast::sqrt(M_PI / (Scalar(2.0) * pair_temp * nu_ij))
                    * exp(du * du / (Scalar(2.0) * pair_temp * nu_ij))
                    * erfc(du / fast::sqrt(Scalar(2.0) * pair_temp * nu_ij));
            f = w * kappa / du;
            }

        factor_f = f;
        factor_r = fast::sqrt(D);
        }

#if !defined(__HIPCC__) && !defined(__METAL_VERSION__)
    static std::string getName()
        {
        return "FrictionLJCoulombNewton";
        }
#endif
    };

typedef EvaluatorPairFrictionLJBase<FrictionLJLinearLaw> EvaluatorPairFrictionLJLinear;
typedef EvaluatorPairFrictionLJBase<FrictionLJCoulombLaw> EvaluatorPairFrictionLJCoulomb;
typedef EvaluatorPairFrictionLJBase<FrictionLJCoulombNewtonLaw>
    EvaluatorPairFrictionLJCoulombNewton;

    } // namespace md
    } // namespace hoomd

#undef DEVICE
#undef HOSTDEVICE

#endif // __PAIR_EVALUATOR_FRICTIONLJVARIANTS_H__
