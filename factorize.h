/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Jan 16, 2021
*******************************************************************/

#pragma once

#include <cstdint>
#include <gmp.h>
#include <vector>

struct FactorInfo
{
    mpz_t m_factor;
    uint64_t m_exp;

    FactorInfo(mpz_srcptr factor, uint64_t exp) : m_exp(exp)
    {
        mpz_init_set(m_factor, factor);
    }

    FactorInfo(uint64_t factor, uint64_t exp) : m_exp(exp)
    {
        mpz_init_set_ui(m_factor, factor);
    }
};

// Factorize n.
//
// Caller assumes ownership of the mpz_t's returned by this function.
// Caller is responsible for mpz_clear'ing them when done with them.
//
// numThreads: 0 to autoselect based on hardware (default and recommended)
//             1 to disable multithreading
std::vector<FactorInfo> Factorize(mpz_srcptr n, uint32_t numThreads = 0);
