// Copyright (c) 2026 Aldo Vásquez-Briceño.
// Part of HOOMD-metal, released under the MIT License (see LICENSE-MIT).
// The Philox4x32-10 generator follows openrand::Philox from OpenRAND, Copyright (c) 2023,
// Michigan State University, also released under the MIT License.

/*! \file Philox.h
    \brief Philox4x32-10 for the Metal Shading Language.

    Produces the same streams as openrand::Philox (hoomd/extern/OpenRAND) for the operations that
    hoomd::RandomGenerator uses, so random numbers on Apple GPUs match the CPU, CUDA, and HIP
    builds bit for bit. OpenRAND itself relies on C++17 and double precision, which the Metal
    Shading Language lacks.
*/

#pragma once

#ifndef __METAL_VERSION__
#error hoomd/mtl/Philox.h is Metal Shading Language only.
#endif

namespace hoomd
    {
namespace mtl
    {
class Philox
    {
    public:
    /*! Same arguments as openrand::Philox:
        \param seed 64-bit key
        \param ctr first counter word
        \param global_seed third counter word
        \param ctr1 second counter word
    */
    Philox(uint64_t seed, uint32_t ctr, uint32_t global_seed, uint32_t ctr1)
        : seed_hi(uint32_t(seed >> 32)), seed_lo(uint32_t(seed & 0xFFFFFFFF)), ctr0(ctr),
          ctr1(ctr1), ctr2(global_seed), m_ctr(0)
        {
        }

    //! Equivalent of openrand::Philox::rand<uint32_t>()
    uint32_t rand_u32()
        {
        generate();
        return m_out[0];
        }

    //! Equivalent of openrand::Philox::rand<uint64_t>()
    uint64_t rand_u64()
        {
        generate();
        return (uint64_t(m_out[0]) << 32) | uint64_t(m_out[1]);
        }

    private:
    void generate()
        {
        uint32_t key0 = seed_hi;
        uint32_t key1 = seed_lo;
        m_out[0] = ctr0;
        m_out[1] = ctr1;
        m_out[2] = ctr2;
        m_out[3] = m_ctr;
        for (int r = 0; r < 10; r++)
            {
            if (r > 0)
                {
                key0 += 0x9E3779B9u;
                key1 += 0xBB67AE85u;
                }
            uint32_t lo0 = 0xD2511F53u * m_out[0];
            uint32_t hi0 = mulhi(0xD2511F53u, m_out[0]);
            uint32_t lo1 = 0xCD9E8D57u * m_out[2];
            uint32_t hi1 = mulhi(0xCD9E8D57u, m_out[2]);
            uint32_t c1 = m_out[1];
            uint32_t c3 = m_out[3];
            m_out[0] = hi1 ^ c1 ^ key0;
            m_out[1] = lo1;
            m_out[2] = hi0 ^ c3 ^ key1;
            m_out[3] = lo0;
            }
        m_ctr++;
        }

    uint32_t seed_hi, seed_lo;
    uint32_t ctr0, ctr1, ctr2;
    uint32_t m_ctr;
    uint32_t m_out[4];
    };
    } // end namespace mtl
    } // end namespace hoomd
