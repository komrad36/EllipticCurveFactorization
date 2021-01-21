/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Jan 16, 2021
*******************************************************************/

#include <algorithm>
#include <atomic>
#include <cmath>
#include <immintrin.h>
#include <thread>

#include "factorize.h"

using I32 = int32_t;
using I64 = int64_t;
using U32 = uint32_t;
using U64 = uint64_t;

static constexpr U32 kSmallPrimes[] =
{
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,
    293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,
    641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,
    1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,
    1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,
    1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,
    1979,1987,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,
    2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,
    2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,
    2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,
    3343,3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,
    3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,4007,4013,4019,4021,
    4027,4049,4051,4057,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297,4327,4337,4339,4349,4357,4363,4373,4391,4397,
    4409,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,4759,4783,
    4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,
    5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,
    5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857,5861,
    5867,5869,5879,5881,5897,5903,5923,5927,5939,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221,6229,6247,6257,6263,
    6269,6271,6277,6287,6299,6301,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571,6577,6581,6599,6607,6619,6637,6653,
    6659,6661,6673,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997,
    7001,7013,7019,7027,7039,7043,7057,7069,7079,7103,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411,7417,7433,7451,
    7457,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,
    7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919,7927,7933,7937,7949,7951,7963,7993,8009,8011,8017,8039,8053,8059,8069,8081,8087,8089,8093,8101,8111,8117,8123,8147,8161,8167,8171,8179,8191,8209,
    8219,8221,8231,8233,8237,8243,8263,8269,8273,8287,8291,8293,8297,8311,8317,8329,8353,8363,8369,8377,8387,8389,8419,8423,8429,8431,8443,8447,8461,8467,8501,8513,8521,8527,8537,8539,8543,8563,8573,8581,8597,8599,8609,
    8623,8627,8629,8641,8647,8663,8669,8677,8681,8689,8693,8699,8707,8713,8719,8731,8737,8741,8747,8753,8761,8779,8783,8803,8807,8819,8821,8831,8837,8839,8849,8861,8863,8867,8887,8893,8923,8929,8933,8941,8951,8963,8969,
    8971,8999,9001,9007,9011,9013,9029,9041,9043,9049,9059,9067,9091,9103,9109,9127,9133,9137,9151,9157,9161,9173,9181,9187,9199,9203,9209,9221,9227,9239,9241,9257,9277,9281,9283,9293,9311,9319,9323,9337,9341,9343,9349,
    9371,9377,9391,9397,9403,9413,9419,9421,9431,9433,9437,9439,9461,9463,9467,9473,9479,9491,9497,9511,9521,9533,9539,9547,9551,9587,9601,9613,9619,9623,9629,9631,9643,9649,9661,9677,9679,9689,9697,9719,9721,9733,9739,
    9743,9749,9767,9769,9781,9787,9791,9803,9811,9817,9829,9833,9839,9851,9857,9859,9871,9883,9887,9901,9907,9923,9929,9931,9941,9949,9967,9973
};

//#define ENABLE_ASSERTS

#if defined(ENABLE_ASSERTS)
#include <cstdio>
#define ASSERT(a) do { if (!(a)) { printf("ASSERTION FAILED: %s\n", #a); __debugbreak(); } } while (0)
#else
#define ASSERT(a) do {} while (0)
#endif

#define ARRAY_COUNT(x) (sizeof((x))/sizeof((x)[0]))

#if !defined(__clang__) && !defined(__GNUC__)
#define UNREACHABLE() __assume(0)
#else
#define UNREACHABLE() __builtin_unreachable()
#endif

static constexpr U32 kMaxTrialDiv = 70000;
static constexpr U64 kMaxTrialDivSqr = U64(kMaxTrialDiv) * U64(kMaxTrialDiv);
static constexpr U32 kSieveSize = 2 * 3 * 5 * 7;
static constexpr U32 kNumRoots = (2 - 1) * (3 - 1) * (5 - 1) * (7 - 1);
static constexpr U32 kSieveUnroll = 16;
static constexpr U32 kNumSieveBlocks = (kSieveUnroll * kSieveSize + 63) >> 6;
static constexpr U32 kLastSmallPrime = kSmallPrimes[ARRAY_COUNT(kSmallPrimes) - 1];
static constexpr bool kLastPrimeAdd2Div3 = (kLastSmallPrime + 2) % 3 == 0;
static constexpr U32 kCheckGap1 = kLastPrimeAdd2Div3 ? 2 : 4;
static constexpr U32 kCheckGap2 = kLastPrimeAdd2Div3 ? 4 : 2;
static constexpr U32 kNextCheckVal = kLastSmallPrime + kCheckGap2;
static constexpr U32 kMtThresh = 4;

static constexpr inline void Swap(U32& a, U32& b)
{
    const U32 t = a;
    a = b;
    b = t;
}

static constexpr inline void Swap(I32& a, I32& b)
{
    const I32 t = a;
    a = b;
    b = t;
}

static inline U32 ToU32(mpz_srcptr x)
{
    ASSERT(mpz_fits_uint_p(x));
    return (U32)mpz_get_ui(x);
}

static bool FitsU64(mpz_srcptr x)
{
    __GMPZ_FITS_UTYPE_P(x, ~0ULL);
}

static inline U64 ToU64(mpz_srcptr x)
{
    ASSERT(FitsU64(x));
    return mpz_get_ui(x);
}

static inline void InvModChecked(mpz_ptr ret, mpz_srcptr x, mpz_srcptr mod)
{
    ASSERT(mpz_cmp_ui(mod, 0) != 0);
    const bool hasInverse = mpz_invert(ret, x, mod);
    ASSERT(hasInverse);
}

static inline void mpz_add_si(mpz_ptr r, mpz_srcptr a, I64 b)
{
    if (b >= 0)
        mpz_add_ui(r, a, (U64)b);
    else
        mpz_sub_ui(r, a, ~((U64)b - 1));
}

static inline U64 CountTrailingZeros(mpz_srcptr x)
{
    ASSERT(mpz_cmp_ui(x, 0) != 0);
    return mpz_scan1(x, 0);
}

static inline double Log2(mpz_srcptr x)
{
    // x = d * 2 ^ e
    // log2(x) = log2(d) + e
    signed long e;
    const double d = mpz_get_d_2exp(&e, x);
    return ::log2(d) + (double)e;
}

struct Mont
{
    void DeepCopyFrom(const Mont& other)
    {
        mpz_set(m_x, other.m_x);
    }

    mpz_t m_x;
};

class MontgomerySystem
{
public:
    MontgomerySystem(mpz_srcptr n)
    {
        ASSERT(mpz_cmp_ui(n, 1) > 0 && mpz_tstbit(n, 0));
        mpz_init_set(m_n, n);
        mpz_init(m_nPrime);
        mpz_init(m_r);
        mpz_init(m_r2);
        mpz_init(m_r3);
        mpz_init(m_scratch);
        mpz_init(m_scratch2);
        m_p = (mpz_sizeinbase(n, 2) + 63) & ~63ULL;
        mpz_setbit(m_scratch, m_p);
        mpz_gcdext(m_r3, m_r2, m_nPrime, m_scratch, n);
        mpz_neg(m_nPrime, m_nPrime);
        if (mpz_cmp_ui(m_nPrime, 0) < 0)
            mpz_add(m_nPrime, m_nPrime, m_scratch);
        ASSERT(mpz_cmp_ui(m_nPrime, 0) >= 0 && mpz_cmp(m_nPrime, m_scratch) < 0);
        mpz_mul(m_scratch, m_scratch, m_scratch);
        mpz_mod(m_r2, m_scratch, n);
        REDC(m_r, m_r2);
        mpz_mul(m_scratch, m_r2, m_r2);
        REDC(m_r3, m_scratch);
    }

    ~MontgomerySystem()
    {
        mpz_clear(m_n);
        mpz_clear(m_nPrime);
        mpz_clear(m_r);
        mpz_clear(m_r2);
        mpz_clear(m_r3);
        mpz_clear(m_scratch);
        mpz_clear(m_scratch2);
    }

    void Add(Mont& r, const Mont& x, const Mont& y) const
    {
        mpz_add(r.m_x, x.m_x, y.m_x);
        if (mpz_cmp(r.m_x, m_n) >= 0)
            mpz_sub(r.m_x, r.m_x, m_n);
    }

    void Sub(Mont& r, const Mont& x, const Mont& y) const
    {
        mpz_sub(r.m_x, x.m_x, y.m_x);
        if (mpz_cmp_ui(r.m_x, 0) < 0)
            mpz_add(r.m_x, r.m_x, m_n);
    }

    void Mul(Mont& r, const Mont& x, const Mont& y) const
    {
        mpz_mul(r.m_x, x.m_x, y.m_x);
        REDC(r.m_x, r.m_x);
    }

    bool InvMod(Mont& r, const Mont& x) const
    {
        if (!mpz_invert(r.m_x, x.m_x, m_n))
            return false;

        mpz_mul(r.m_x, r.m_x, m_r3);
        REDC(r.m_x, r.m_x);
        return true;
    }

    void Sqr(Mont& r, const Mont& x) const
    {
        Mul(r, x, x);
    }

    void GcdN(mpz_ptr r, const Mont& x) const
    {
        mpz_gcd(r, x.m_x, m_n);
    }

    void ToMontgomery(Mont& r, mpz_srcptr x) const
    {
        mpz_mod(r.m_x, x, m_n);
        mpz_mul(r.m_x, r.m_x, m_r2);
        REDC(r.m_x, r.m_x);
    }

    void MakeOne(Mont& r) const
    {
        mpz_set(r.m_x, m_r);
    }

    bool IsZero(Mont& x) const
    {
        return mpz_cmp_ui(x.m_x, 0) == 0;
    }

private:
    inline void REDC(mpz_ptr r, mpz_srcptr T) const
    {
        mpz_mod_2exp(m_scratch, T, m_p);
        mpz_mul(m_scratch2, m_scratch, m_nPrime);
        mpz_mod_2exp(m_scratch2, m_scratch2, m_p);
        mpz_mul(m_scratch, m_scratch2, m_n);
        mpz_add(m_scratch, m_scratch, T);
        mpz_div_2exp(r, m_scratch, m_p);
        if (mpz_cmp(r, m_n) >= 0)
            mpz_sub(r, r, m_n);
    }

    mpz_t m_n;
    mpz_t m_nPrime;
    mpz_t m_r;
    mpz_t m_r2;
    mpz_t m_r3;
    mutable mpz_t m_scratch;
    mutable mpz_t m_scratch2;
    U64 m_p;
};

static inline void Swap(Mont& a, Mont& b)
{
    mpz_swap(a.m_x, b.m_x);
}

static inline void Rot(Mont& a, Mont& b, Mont& c)
{
    mpz_swap(a.m_x, b.m_x);
    mpz_swap(b.m_x, c.m_x);
}

class MpzJanitor
{
    mpz_t& m_x;

public:
    MpzJanitor(mpz_t& x) : m_x(x)
    {
        mpz_init(x);
    }
    ~MpzJanitor()
    {
        mpz_clear(m_x);
    }
};

class MontJanitor
{
    Mont& m_x;

public:
    MontJanitor(Mont& x) : m_x(x)
    {
        mpz_init(x.m_x);
    }
    ~MontJanitor()
    {
        mpz_clear(m_x.m_x);
    }
};

class MontArrayJanitor
{
    Mont* m_v;
    U32 m_n;

public:
    MontArrayJanitor(Mont* v, U32 n) : m_v(v), m_n(n)
    {
        for (U32 i = 0; i < n; ++i)
            mpz_init(v[i].m_x);
    }
    ~MontArrayJanitor()
    {
        for (U32 i = 0; i < m_n; ++i)
            mpz_clear(m_v[i].m_x);
    }
};

#define MPZ_CREATE(x) mpz_t x; MpzJanitor mpzJanitor##x((x))
#define MONT_CREATE(x) Mont x; MontJanitor montJanitor##x((x))
#define MONT_ARRAY_CREATE(x, n) Mont x[(n)]; MontArrayJanitor montArrayJanitor##x((x),(n))

#define PROCESS_FACTOR()                                \
do                                                      \
{                                                       \
    if (FitsU64(x) && U64(p) * U64(p) > ToU64(x))       \
        return true;                                    \
    if (mpz_divisible_ui_p(x, p))                       \
    {                                                   \
        mpz_set_ui(P, p);                               \
        const U64 e = mpz_remove(x, x, P);              \
        ret.emplace_back(p, e);                         \
    }                                                   \
} while (0)

static bool TrialDiv(std::vector<FactorInfo>& ret, mpz_ptr x)
{
    MPZ_CREATE(P);

    // use small primes
    for (U32 i = 1; i < ARRAY_COUNT(kSmallPrimes); ++i)
    {
        const U32 p = kSmallPrimes[i];
        PROCESS_FACTOR();
    }

    for (U64 p = kNextCheckVal; p <= kMaxTrialDiv;)
    {
        PROCESS_FACTOR();
        p += kCheckGap1;
        PROCESS_FACTOR();
        p += kCheckGap2;
    }

    return false;
}

#undef PROCESS_FACTOR

static void Sort(std::vector<FactorInfo>& v)
{
    if (v.size() < 2)
        return;

    std::sort(v.begin(), v.end(), [](const FactorInfo& a, const FactorInfo& b) { return mpz_cmp(a.m_factor, b.m_factor) < 0; });

    std::vector<FactorInfo> newV{v[0]};
    for (U64 i = 1; i < v.size(); ++i)
    {
        if (mpz_cmp(newV.back().m_factor, v[i].m_factor) != 0)
        {
            newV.emplace_back(v[i]);
        }
        else
        {
            newV.back().m_exp += v[i].m_exp;
            mpz_clear(v[i].m_factor);
        }
    }
    v = newV;
}

static bool RecordFactor(std::vector<FactorInfo>& v, mpz_srcptr f)
{
    MPZ_CREATE(g);

    bool updated = false;

    const U64 numFactors = v.size();
    for (U64 i = 0; i < numFactors; ++i)
    {
        FactorInfo& info = v[i];
        mpz_gcd(g, info.m_factor, f);
        if (mpz_cmp_ui(g, 2) < 0 || mpz_cmp(g, info.m_factor) == 0)
            continue;

        updated = true;
        ASSERT(mpz_divisible_p(info.m_factor, g));
        mpz_divexact(info.m_factor, info.m_factor, g);
        v.emplace_back(g, info.m_exp);
    }

    if (updated)
        Sort(v);

    return updated;
}

static void RecordPrime(std::vector<FactorInfo>& ret, std::vector<FactorInfo>& v, U64 i)
{
    for (FactorInfo& old : ret)
    {
        if (mpz_cmp(old.m_factor, v[i].m_factor) == 0)
        {
            old.m_exp += v[i].m_exp;
            mpz_clear(v[i].m_factor);
            goto label_found;
        }
    }

    ret.emplace_back(v[i]);

    label_found:;

    if (i != v.size() - 1)
        v[i] = v.back();
    v.pop_back();
}

#define PROCESS_EULER_TOTIENT(p)                        \
do                                                      \
{                                                       \
    if (U32(p)*U32(p) > x)                              \
        return x > 1U ? r * (U32(x) - 1U) / U32(x) : r; \
    if (!(x % U32(p)))                                  \
    {                                                   \
        do                                              \
        {                                               \
            x /= U32(p);                                \
        } while (!(x % U32(p)));                        \
        r = r * (U32(p) - 1U) / U32(p);                 \
    }                                                   \
} while (0)

static constexpr U32 EulerTotient(U32 x)
{
    ASSERT(29 * 29 > x);

    U32 r = x;
    PROCESS_EULER_TOTIENT(2);
    PROCESS_EULER_TOTIENT(3);
    PROCESS_EULER_TOTIENT(5);
    PROCESS_EULER_TOTIENT(7);
    PROCESS_EULER_TOTIENT(11);
    PROCESS_EULER_TOTIENT(13);
    PROCESS_EULER_TOTIENT(17);
    PROCESS_EULER_TOTIENT(19);
    PROCESS_EULER_TOTIENT(23);
    PROCESS_EULER_TOTIENT(29);
    UNREACHABLE();
}

#undef PROCESS_EULER_TOTIENT

#define PROCESS_MOBIUS(p)               \
do                                      \
{                                       \
    if (U32(p)*U32(p) > x)              \
        return x > 1U ? -r : r;         \
    if (!(x % U32(p)))                  \
    {                                   \
        x /= U32(p);                    \
        if (!(x % U32(p)))              \
            return 0;                   \
        r = -r;                         \
    }                                   \
} while (0)

static constexpr I32 Mobius(U32 x)
{
    ASSERT(29 * 29 > x);

    I32 r = 1;
    PROCESS_MOBIUS(2);
    PROCESS_MOBIUS(3);
    PROCESS_MOBIUS(5);
    PROCESS_MOBIUS(7);
    PROCESS_MOBIUS(11);
    PROCESS_MOBIUS(13);
    PROCESS_MOBIUS(17);
    PROCESS_MOBIUS(19);
    PROCESS_MOBIUS(23);
    PROCESS_MOBIUS(29);
    UNREACHABLE();
}

#undef PROCESS_MOBIUS

static constexpr I32 JacobiSymbol(I32 n, I32 k)
{
    ASSERT(k > 0 && (k & 1));
    n %= k;
    I32 t = 1;
    while (n)
    {
        while (!(n & 1))
        {
            n >>= 1;
            if (!(((k & 7) - 3) & -3))
                t = -t;
        }
        Swap(n, k);
        if ((n & k & 3) == 3)
            t = -t;
        n %= k;
    }
    return k == 1 ? t : 0;
}

static U32 CountTrailingZeros(U32 x)
{
    return (U32)_tzcnt_u32(x);
}

static U32 CountTrailingZeros(U64 x)
{
    return (U32)_tzcnt_u64(x);
}

static U32 Gcd(U32 a, U32 b)
{
    if (!a)
        return b;
    if (!b)
        return a;

    const U32 za = CountTrailingZeros(a);
    a >>= za;
    const U32 zb = CountTrailingZeros(b);
    b >>= zb;

    const U32 m = za < zb ? za : zb;
    while (a != b)
    {
        if (a > b)
        {
            a -= b;
            a >>= CountTrailingZeros(a);
        }
        else
        {
            b -= a;
            b >>= CountTrailingZeros(b);
        }
    }

    return a << m;
}

static bool Aurifeuille(std::vector<FactorInfo>& v, mpz_srcptr b, U64 e, I32 ofs)
{
    MPZ_CREATE(x);
    MPZ_CREATE(C);
    MPZ_CREATE(D);

    ASSERT(mpz_cmp_ui(b, 0) > 0);
    ASSERT(e);

    constexpr U32 kMaxBase = 385;

    if (mpz_cmp_ui(b, kMaxBase) > 0)
        return false;

    const U32 n = ToU32(b);

    if (!(e & 1) && ofs == -1)
    {
        e >>= CountTrailingZeros(e);
        ofs = (I32)(n & 3) - 2;
    }

    if ((e % n) || !((e / n) & 1) || (((n & 3) == 1 || ofs != 1) && ((n & 3) != 1 || ofs != -1)))
        return false;

    bool updated = false;

    const U32 n1 = (n & 3) == 1 ? n : n << 1;
    const U32 d = EulerTotient(n1) >> 1;

    I64 coef[kMaxBase + 2];
    coef[0] = 0;

    for (U32 k = 1; k - 1 <= d; k += 2)
        coef[k] = JacobiSymbol((I32)n, (I32)k);

    for (U32 k = 2; k - 1 <= d; k += 2)
    {
        coef[k] = 0;
        const U32 r = ((n - 1) * k) & 7;
        if (r == 0 || r == 4)
        {
            const U32 gcd = Gcd(k, n1);
            coef[k] = (I64)Mobius(n1 / gcd) * (I64)EulerTotient(gcd);
            if (r == 4)
                coef[k] = -coef[k];
        }
    }

    I64 Cdp[kMaxBase + 1];
    I64 Ddp[kMaxBase + 1];
    Cdp[0] = Ddp[0] = 1;
    for (U32 k = 1; k <= d >> 1; ++k)
    {
        Cdp[k] = Ddp[k] = 0;
        for (U32 j = 0; j < k; ++j)
        {
            const U32 m = (k << 1) - (j << 1);
            ASSERT(m > 0 && m < kMaxBase);
            Cdp[k] = Cdp[k] + I64(n) * coef[m - 1] * Ddp[j] - coef[m] * Cdp[j];
            Ddp[k] = Ddp[k] +          coef[m + 1] * Cdp[j] - coef[m] * Ddp[j];
        }
        Cdp[k] /= I64(k) << 1;
        Ddp[k] = (Ddp[k] + Cdp[k]) / ((I64(k) << 1) + 1);
    }

    for (U32 k = (d >> 1) + 1; k <= d; ++k)
        Cdp[k] = Cdp[d - k];

    for (U32 k = (d >> 1) + (d & 1); k < d; ++k)
        Ddp[k] = Ddp[d - 1 - k];

    const U64 q = e / n;
    for (U64 L = 1; L * L <= q; L += 2)
    {
        if (!(q % L))
        {
            const U64 qDivL = q / L;
            for (U32 i = 0; i <= U32(qDivL != L); ++i)
            {
                const U64 r = i ? qDivL : L;
                mpz_pow_ui(x, b, r);
                mpz_set_ui(C, 1);
                mpz_set_ui(D, 1);
                for (U32 k = 1; k < d; ++k)
                {
                    mpz_mul(C, C, x);
                    mpz_add_si(C, C, Cdp[k]);
                    mpz_mul(D, D, x);
                    mpz_add_si(D, D, Ddp[k]);
                }
                mpz_mul(C, C, x);
                mpz_add_si(C, C, Cdp[d]);
                mpz_pow_ui(x, b, (r + 1) >> 1);
                mpz_mul(D, D, x);
                mpz_add(x, C, D);
                mpz_sub(x, C, D);
                if (RecordFactor(v, x))
                    updated = true;
                if (RecordFactor(v, x))
                    updated = true;
            }
        }
    }

    return updated;
}

enum class BpswResult
{
    kProbablePrime,
    kComposite,
    kCompositeFermatPseudoprime,
};

static BpswResult BpswPrimalityTest(mpz_srcptr x)
{
    MPZ_CREATE(q);
    MPZ_CREATE(m);
    MPZ_CREATE(D);
    MPZ_CREATE(U);
    MPZ_CREATE(V);
    MPZ_CREATE(Q);
    MPZ_CREATE(t);
    MPZ_CREATE(newU);

    ASSERT(mpz_cmp_ui(x, 2) > 0);
    ASSERT(mpz_tstbit(x, 0));

    // 2-Fermat and 2-SPRP tests
    {
        mpz_sub_ui(q, x, 1);
        const U64 e = CountTrailingZeros(q);
        mpz_div_2exp(q, q, e);
        mpz_set_ui(m, 2);
        mpz_powm(q, m, q, x);
        mpz_sub_ui(m, x, 1);
        if (mpz_cmp_ui(q, 1) != 0 && mpz_cmp(q, m) != 0)
        {
            for (U64 i = 0; i < e - 1; ++i)
            {
                mpz_mul(q, q, q);
                mpz_mod(q, q, x);
                if (mpz_cmp_ui(q, 1) == 0)
                    return BpswResult::kCompositeFermatPseudoprime;
                if (mpz_cmp(q, m) == 0)
                    goto label_sprp;
            }

            return BpswResult::kComposite;
        }
    }

    label_sprp:;

    if (mpz_perfect_square_p(x))
        return BpswResult::kCompositeFermatPseudoprime;

    // strong lucas test
    I32 iD = 5;
    {
        bool negD = false;
        for (;;)
        {
            mpz_mod_ui(m, x, (U32)iD);
            if (JacobiSymbol((I32)ToU32(m), iD) == -1)
                break;
            negD = !negD;
            iD += 2;
        }
        if (negD)
            iD = -iD;
    }

    const I64 iQ = (1LL - (I64)iD) >> 2;

    mpz_set_si(D, iD);
    mpz_set_si(m, iQ);
    mpz_set_ui(U, 0);
    mpz_set_ui(V, 2);
    mpz_set_ui(Q, 1);

    mpz_add_ui(t, x, 1);
    const U64 e = CountTrailingZeros(t);
    mpz_div_2exp(t, t, e);

    for (U64 i = mpz_sizeinbase(t, 2); i-- > 0;)
    {
        mpz_mul(U, U, V);
        mpz_mod(U, U, x);
        mpz_mul(V, V, V);
        mpz_sub(V, V, Q);
        mpz_sub(V, V, Q);
        mpz_mod(V, V, x);
        mpz_mul(Q, Q, Q);
        mpz_mod(Q, Q, x);

        if (mpz_tstbit(t, i))
        {
            ASSERT(mpz_cmp_ui(U, 0) >= 0 && mpz_cmp(U, x) < 0);
            ASSERT(mpz_cmp_ui(V, 0) >= 0 && mpz_cmp(V, x) < 0);
            mpz_add(newU, U, V);

            if (!mpz_tstbit(newU, 0))
            {
                mpz_div_2exp(newU, newU, 1);
            }
            else if (mpz_cmp(newU, x) >= 0)
            {
                mpz_sub(newU, newU, x);
                ASSERT(!mpz_tstbit(newU, 0));
                mpz_div_2exp(newU, newU, 1);
            }
            else
            {
                mpz_add(newU, newU, x);
                ASSERT(!mpz_tstbit(newU, 0));
                mpz_div_2exp(newU, newU, 1);
            }

            mpz_addmul(V, D, U);
            mpz_mod(V, V, x);
            if (mpz_tstbit(V, 0))
                mpz_add(V, V, x);
            mpz_div_2exp(V, V, 1);
            mpz_set(U, newU);
            mpz_mul(Q, Q, m);
            mpz_mod(Q, Q, x);
        }
    }

    if (mpz_cmp_ui(U, 0) == 0)
        return BpswResult::kProbablePrime;

    for (U64 i = 0; i < e; ++i)
    {
        if (mpz_cmp_ui(V, 0) == 0)
            return BpswResult::kProbablePrime;

        mpz_mul(V, V, V);
        mpz_sub(V, V, Q);
        mpz_sub(V, V, Q);
        mpz_mod(V, V, x);
        mpz_mul(Q, Q, Q);
        mpz_mod(Q, Q, x);
    }

    return BpswResult::kCompositeFermatPseudoprime;
}

static bool Fermat(std::vector<FactorInfo>& v, mpz_srcptr x)
{
    MPZ_CREATE(q);
    MPZ_CREATE(xSub1);
    MPZ_CREATE(sqrtOne);
    MPZ_CREATE(sqrtNegOne);
    MPZ_CREATE(B);
    MPZ_CREATE(r);
    MPZ_CREATE(r2);
    MPZ_CREATE(g);

    ASSERT(mpz_tstbit(x, 0));
    ASSERT(mpz_cmp_ui(x, 1) > 0);

    mpz_sub_ui(xSub1, x, 1);
    mpz_set(q, xSub1);
    const U64 e = CountTrailingZeros(q);
    mpz_div_2exp(q, q, e);

    bool sqrtOneFound = false;
    bool sqrtNegOneFound = false;
    bool updated = false;
    U32 b = 0;
    for (U32 iIter = 0; iIter < 16; ++iIter)
    {
        b ^= 0x9E3779B9U + (b << 6) + (b >> 2);
        mpz_set_ui(B, b);
        mpz_powm(r, B, q, x);
        if (mpz_cmp_ui(r, 1) == 0 || mpz_cmp(r, xSub1) == 0)
            continue;

        for (U64 i = 0; i < e; ++i)
        {
            mpz_mul(r2, r, r);
            mpz_mod(r2, r2, x);

            if (mpz_cmp_ui(r2, 1) == 0)
            {
                if (sqrtOneFound)
                {
                    mpz_sub(g, r, sqrtOne);
                    mpz_mod(g, g, x);
                    mpz_gcd(g, g, x);
                    if (mpz_cmp_ui(g, 1) != 0 && mpz_cmp(g, x) != 0)
                    {
                        if (RecordFactor(v, g))
                            updated = true;
                    }
                }
                else
                {
                    mpz_set(sqrtOne, r);
                    sqrtOneFound = true;
                }

                mpz_add_ui(g, r, 1);
                mpz_mod(g, g, x);
                mpz_gcd(g, g, x);
                if (mpz_cmp_ui(g, 1) != 0 && mpz_cmp(g, x) != 0)
                {
                    if (RecordFactor(v, g))
                        updated = true;
                }

                break;
            }

            if (mpz_cmp(r2, xSub1) == 0)
            {
                if (sqrtNegOneFound)
                {
                    mpz_sub(g, r, sqrtNegOne);
                    mpz_mod(g, g, x);
                    mpz_gcd(g, g, x);
                    if (mpz_cmp_ui(g, 1) != 0 && mpz_cmp(g, x) != 0)
                    {
                        if (RecordFactor(v, g))
                            updated = true;
                    }
                }
                else
                {
                    mpz_set(sqrtNegOne, r);
                    sqrtNegOneFound = true;
                }

                break;
            }

            mpz_set(r, r2);
        }
    }

    return updated;
}

static constexpr U64 PerfectSqrMask(U32 p)
{
    U64 r = 1;
    for (U32 a = 1; a < p; ++a)
        r |= (U64(JacobiSymbol((I32)a, (I32)p) == 1) << a);
    return r;
}

static bool Lehmann(mpz_ptr ret, mpz_srcptr x, U32 k)
{
    MPZ_CREATE(sqr);
    MPZ_CREATE(a);
    MPZ_CREATE(b);
    MPZ_CREATE(c);
    MPZ_CREATE(d);

    // bit i of element corresponding to prime p: i is perfect square (mod p)
    static constexpr U64 kPerfectSqr[] =
    {
        PerfectSqrMask( 3),
        PerfectSqrMask( 5),
        PerfectSqrMask( 7),
        PerfectSqrMask(11),
        PerfectSqrMask(13),
        PerfectSqrMask(17),
        PerfectSqrMask(19),
        PerfectSqrMask(23),
        PerfectSqrMask(29),
        PerfectSqrMask(31),
        PerfectSqrMask(37),
        PerfectSqrMask(41),
        PerfectSqrMask(43),
        PerfectSqrMask(47),
        PerfectSqrMask(53),
        PerfectSqrMask(59),
        PerfectSqrMask(61),
    };

    constexpr U32 kNumLehmannPrimes = ARRAY_COUNT(kPerfectSqr);

    U32 m, r, s;
    if (!mpz_tstbit(x, 0))
    {
        r = 0;
        m = 1;
        s = 0;
    }
    else if (k & 1)
    {
        mpz_add_ui(a, x, k);
        mpz_mod_2exp(a, a, 2);
        r = ToU32(a);
        m = 4;
        s = 2;
    }
    else
    {
        r = 1;
        m = 2;
        s = 1;
    }

    mpz_set_ui(sqr, k);
    mpz_mul_2exp(sqr, sqr, 2);
    mpz_mul(sqr, sqr, x);
    mpz_sqrt(a, sqr);

    for (;;)
    {
        mpz_mod_2exp(b, a, s);
        if (mpz_cmp_ui(b, r) == 0)
        {
            mpz_mul(b, a, a);
            if (mpz_cmp(b, sqr) >= 0)
                break;
        }
        mpz_add_ui(a, a, 1);
    }

    mpz_mul(d, a, a);
    mpz_sub(d, d, sqr);
    ASSERT(mpz_cmp_ui(d, 0) >= 0);

    U32 v[kNumLehmannPrimes];
    U32 deltas[kNumLehmannPrimes];
    for (U32 i = 0; i < kNumLehmannPrimes; ++i)
    {
        const U32 p = kSmallPrimes[i + 1];
        mpz_mod_ui(c, d, p);
        v[i] = ToU32(c);
        mpz_mod_ui(c, a, p);
        deltas[i] = (m * ((ToU32(c) << 1) + m)) % p;
    }

    for (U32 i = 0; i < 12000; ++i)
    {
        for (U32 j = 0; j < kNumLehmannPrimes; ++j)
        {
            if (!(kPerfectSqr[j] & (1ULL << v[j])))
                goto label_not_perfect_sqr;
        }

        mpz_add_ui(c, a, m * i);
        mpz_mul(d, c, c);
        mpz_sub(d, d, sqr);
        mpz_sqrt(d, d);
        mpz_add(d, d, c);
        mpz_gcd(ret, d, x);
        if (mpz_cmp_ui(ret, kMaxTrialDiv) > 0)
            return true;

        label_not_perfect_sqr:;

        for (U32 j = 0; j < kNumLehmannPrimes; ++j)
        {
            v[j] = (v[j] + deltas[j]) % kSmallPrimes[j + 1];
            deltas[j] = (deltas[j] + ((m * m) << 1)) % kSmallPrimes[j + 1];
        }
    }

    return false;
}

static inline void EcMam(const MontgomerySystem& ms, Mont& r, const Mont& a, const Mont& b, const Mont& c)
{
    ms.Mul(r, a, b);
    ms.Add(r, r, c);
    ms.Mul(r, r, a);
}

static void EcAdd(const MontgomerySystem& ms, Mont& x3, Mont& z3, Mont& S1, Mont& S2, Mont& S3, Mont& S4,
    const Mont& x2, const Mont& z2, const Mont& x1, const Mont& z1, const Mont& x, const Mont& z)
{
    ms.Sub(S1, x2, z2);
    ms.Add(S2, x1, z1);
    ms.Mul(S1, S1, S2);
    ms.Sub(S2, x1, z1);
    ms.Add(S3, x2, z2);
    ms.Mul(S2, S2, S3);
    ms.Add(S3, S1, S2);
    ms.Sqr(S3, S3);
    ms.Mul(S3, S3, z);
    ms.Sub(S4, S1, S2);
    ms.Sqr(S4, S4);
    ms.Mul(z3, S4, x);
    x3.DeepCopyFrom(S3);
}

static void EcAddMove(const MontgomerySystem& ms, Mont& x3, Mont& z3, Mont& S1, Mont& S2, Mont& S3, Mont& S4,
    const Mont& x2, const Mont& z2, const Mont& x1, const Mont& z1, const Mont& x, const Mont& z)
{
    ms.Sub(S3, x2, z2);
    ms.Add(S2, x1, z1);
    ms.Mul(S1, S3, S2);
    ms.Sub(S2, x1, z1);
    ms.Add(x3, x2, z2);
    ms.Mul(S3, S2, x3);
    ms.Add(x3, S1, S3);
    ms.Sqr(S4, x3);
    ms.Mul(x3, S4, z);
    ms.Sub(S1, S1, S3);
    ms.Sqr(S4, S1);
    ms.Mul(z3, S4, x);
}

static void EcMul(const MontgomerySystem& ms, Mont& x2, Mont& z2, Mont& S1, Mont& S2,
    const Mont& x1, const Mont& z1, const Mont& b)
{
    ms.Add(S1, x1, z1);
    ms.Sqr(S1, S1);
    ms.Sub(S2, x1, z1);
    ms.Sqr(S2, S2);
    ms.Mul(x2, S1, S2);
    ms.Sub(S1, S1, S2);
    EcMam(ms, z2, S1, b, S2);
}

static void EcPrac(const MontgomerySystem& ms, U32 i, Mont& S1, Mont& S2, Mont& S3, Mont& S4, Mont& S5, Mont& S6,
    Mont& S7, Mont& S8, Mont& S9, Mont& S10, Mont& S11, Mont& S12, Mont& x, Mont& z, const Mont& O)
{
    static constexpr double v[] =
    {
        1.6180339887498948,    // (0    + phi) / 1,
        1.7236067977499790,    // (7    + phi) / 5,
        1.6183471196562281,    // (2311 + phi) / 1429,
        1.6179144065288179,    // (6051 - phi) / 3739,
        1.6124299495094950,    // (129  - phi) / 79,
        1.6328398060887063,    // (49   + phi) / 31,
        1.6201819808074158,    // (337  + phi) / 209,
        1.5801787282954641,    // (19   - phi) / 11,
        1.6172146165344039,    // (883  - phi) / 545,
        1.3819660112501052,    // (3    - phi) / 1,
    };

    U32 bestR = (U32)(i / v[0] + 0.5);
    {
        U32 bestC = ~0U;
        for (const double s : v)
        {
            constexpr U32 kAddCost = 6;
            constexpr U32 kDupCost = 5;

            const U32 r = (U32)(i / s + 0.5);
            U32 c;
            if (r >= i)
            {
                c = kAddCost * i;
            }
            else
            {
                U32 d = i - r;
                U32 e = (r << 1) - i;
                c = kDupCost + kAddCost;
                while (d != e)
                {
                    if (d < e)
                        Swap(d, e);
                    if ((d << 2) <= 5 * e && !((d + e) % 3))
                    {
                        const U32 t = ((d << 1) - e) / 3;
                        e = ((e << 1) - d) / 3;
                        d = t;
                        c += 3 * kAddCost;
                    }
                    else if ((d << 2) <= 5 * e && !((d - e) % 6))
                    {
                        d = (d - e) >> 1;
                        c += kAddCost + kDupCost;
                    }
                    else if (d <= (e << 2))
                    {
                        d -= e;
                        c += kAddCost;
                    }
                    else if (!((d + e) & 1))
                    {
                        d = (d - e) >> 1;
                        c += kAddCost + kDupCost;
                    }
                    else if (!(d & 1))
                    {
                        d >>= 1;
                        c += kAddCost + kDupCost;
                    }
                    else if (!(d % 3))
                    {
                        d = d / 3 - e;
                        c += 3 * kAddCost + kDupCost;
                    }
                    else if (!((d + e) % 3))
                    {
                        d = (d - (e << 1)) / 3;
                        c += 3 * kAddCost + kDupCost;
                    }
                    else if (!((d - e) % 3))
                    {
                        d = (d - e) / 3;
                        c += 3 * kAddCost + kDupCost;
                    }
                    else
                    {
                        ASSERT(!(e & 1));
                        e >>= 1;
                        c += kAddCost + kDupCost;
                    }
                }
                ASSERT(d == 1);
            }

            if (c < bestC)
            {
                bestC = c;
                bestR = r;
            }
        }
    }

    U32 d = i - bestR;
    U32 e = (bestR << 1) - i;
    Mont& xA = x;
    Mont& zA = z;
    Mont& xB = S9;
    Mont& zB = S10;
    Mont& xC = S11;
    Mont& zC = S12;
    xB.DeepCopyFrom(x);
    zB.DeepCopyFrom(z);
    xC.DeepCopyFrom(x);
    zC.DeepCopyFrom(z);
    EcMul(ms, xA, zA, S5, S6, xA, zA, O);
    while (d != e)
    {
        if (d < e)
        {
            Swap(d, e);
            Swap(xA, xB);
            Swap(zA, zB);
        }
        if ((d << 2) <= 5 * e && !((d + e) % 3))
        {
            const U32 t = ((d << 1) - e) / 3;
            e = ((e << 1) - d) / 3;
            d = t;
            EcAddMove(ms, S1, S2, S3, S4, S5, S6, xA, zA, xB, zB, xC, zC);
            EcAddMove(ms, S3, S4, S5, S6, S7, S8, S1, S2, xA, zA, xB, zB);
            EcAddMove(ms, xB, zB, S5, S6, S7, S8, xB, zB, S1, S2, xA, zA);
            Swap(xA, S3);
            Swap(zA, S4);
        }
        else if ((d << 2) <= 5 * e && !((d - e) % 6))
        {
            d = (d - e) >> 1;
            EcAddMove(ms, xB, zB, S5, S6, S7, S8, xA, zA, xB, zB, xC, zC);
            EcMul(ms, xA, zA, S5, S6, xA, zA, O);
        }
        else if (d <= (e << 2))
        {
            d -= e;
            EcAddMove(ms, S1, S2, S5, S6, S7, S8, xB, zB, xA, zA, xC, zC);
            Rot(xB, S1, xC);
            Rot(zB, S2, zC);
        }
        else if (!((d + e) & 1))
        {
            d = (d - e) >> 1;
            EcAddMove(ms, xB, zB, S5, S6, S7, S8, xB, zB, xA, zA, xC, zC);
            EcMul(ms, xA, zA, S5, S6, xA, zA, O);
        }
        else if (!(d & 1))
        {
            d >>= 1;
            EcAddMove(ms, xC, zC, S5, S6, S7, S8, xC, zC, xA, zA, xB, zB);
            EcMul(ms, xA, zA, S5, S6, xA, zA, O);
        }
        else if (!(d % 3))
        {
            d = d / 3 - e;
            EcMul(ms, S1, S2, S5, S6, xA, zA, O);
            EcAddMove(ms, S3, S4, S5, S6, S7, S8, xA, zA, xB, zB, xC, zC);
            EcAdd(ms, xA, zA, S5, S6, S7, S8, S1, S2, xA, zA, xA, zA);
            EcAddMove(ms, S1, S2, S5, S6, S7, S8, S1, S2, S3, S4, xC, zC);
            Rot(xC, xB, S1);
            Rot(zC, zB, S2);
        }
        else if (!((d + e) % 3))
        {
            d = (d - (e << 1)) / 3;
            EcAddMove(ms, S1, S2, S5, S6, S7, S8, xA, zA, xB, zB, xC, zC);
            EcAdd(ms, xB, zB, S5, S6, S7, S8, S1, S2, xA, zA, xB, zB);
            EcMul(ms, S1, S2, S5, S6, xA, zA, O);
            EcAdd(ms, xA, zA, S5, S6, S7, S8, xA, zA, S1, S2, xA, zA);
        }
        else if (!((d - e) % 3))
        {
            d = (d - e) / 3;
            EcAddMove(ms, S1, S2, S5, S6, S7, S8, xA, zA, xB, zB, xC, zC);
            EcAddMove(ms, xC, zC, S5, S6, S7, S8, xC, zC, xA, zA, xB, zB);
            Swap(xB, S1);
            Swap(zB, S2);
            EcMul(ms, S1, S2, S5, S6, xA, zA, O);
            EcAdd(ms, xA, zA, S3, S4, S5, S6, xA, zA, S1, S2, xA, zA);
        }
        else
        {
            ASSERT(!(e & 1));
            e >>= 1;
            EcAddMove(ms, xC, zC, S5, S6, S7, S8, xC, zC, xB, zB, xA, zA);
            EcMul(ms, xB, zB, S5, S6, xB, zB, O);
        }
    }
    EcAddMove(ms, x, z, S5, S6, S7, S8, xA, zA, xB, zB, xC, zC);
    ASSERT(d == 1);
}

static inline constexpr U64 BuildBitMask(U64 i0, U64 i1)
{
    return (~0ULL << (i0 & 63)) & (~0ULL >> ((i1 ^ 63) & 63));
}

static void CopyBits(U64* vDst, U64 iDst, const U64* vSrc, U64 iSrc, U64 numBits)
{
    if (numBits == 0)
        return;

    const U64 iDst0 = iDst;
    const U64 iDst1 = iDst + numBits - 1;

    const U64 iSrc0 = iSrc;
    const U64 iSrc1 = iSrc + numBits - 1;

    const U64 iBlockSrc0 = iSrc0 >> 6;
    const U64 iBlockSrc1 = iSrc1 >> 6;

    const U64 iBlockDst0 = iDst0 >> 6;
    const U64 iBlockDst1 = iDst1 >> 6;

    const U64 ofsRight = (iSrc0 - iDst0) & 63;
    const U64 ofsLeft = (iDst0 - iSrc0) & 63;

    if (iBlockDst0 == iBlockDst1)
    {
        // special case: single dest block

        const U64 blockSrc = (vSrc[iBlockSrc0] >> ofsRight) | (vSrc[iBlockSrc1] << ofsLeft);
        const U64 blockDst = vDst[iBlockDst0];
        const U64 mask = BuildBitMask(iDst0 & 63, iDst1 & 63);
        vDst[iBlockDst0] = (blockDst & ~mask) | (blockSrc & mask);
    }
    else if (ofsRight == 0)
    {
        // src and dest have same alignment

        // handle first dest block
        {
            const U64 blockSrc = vSrc[iBlockSrc0];
            const U64 blockDst = vDst[iBlockDst0];
            const U64 mask = BuildBitMask(iDst0 & 63, 63);
            vDst[iBlockDst0] = (blockDst & ~mask) | (blockSrc & mask);
        }

        // handle middle (whole) dest blocks
        for (U64 iBlockDst = iBlockDst0 + 1, iBlockSrc = iBlockSrc0 + 1; iBlockDst < iBlockDst1; ++iBlockDst, ++iBlockSrc)
        {
            vDst[iBlockDst] = vSrc[iBlockSrc];
        }

        // handle final dest block
        {
            const U64 blockSrc = vSrc[iBlockSrc1];
            const U64 blockDst = vDst[iBlockDst1];
            const U64 mask = BuildBitMask(0, iDst1 & 63);
            vDst[iBlockDst1] = (blockDst & ~mask) | (blockSrc & mask);
        }
    }
    else
    {
        U64 iBlockSrc = iBlockSrc0 + ((iSrc0 & 63) > (iDst0 & 63));

        // handle first dest block
        {
            const U64 blockSrc = (vSrc[iBlockSrc0] >> ofsRight) | (vSrc[iBlockSrc] << ofsLeft);
            const U64 blockDst = vDst[iBlockDst0];
            const U64 mask = BuildBitMask(iDst0 & 63, 63);
            vDst[iBlockDst0] = (blockDst & ~mask) | (blockSrc & mask);
        }

        // handle middle (whole) dest blocks
        for (U64 iBlockDst = iBlockDst0 + 1; iBlockDst < iBlockDst1; ++iBlockDst, ++iBlockSrc)
        {
            vDst[iBlockDst] = (vSrc[iBlockSrc] >> ofsRight) | (vSrc[iBlockSrc + 1] << ofsLeft);
        }

        // handle final dest block
        {
            const U64 blockSrc = (vSrc[iBlockSrc] >> ofsRight) | (vSrc[iBlockSrc1] << ofsLeft);
            const U64 blockDst = vDst[iBlockDst1];
            const U64 mask = BuildBitMask(0, iDst1 & 63);
            vDst[iBlockDst1] = (blockDst & ~mask) | (blockSrc & mask);
        }
    }
}

static void MakeSieve(U64* pSieve, const U64* pPrecomp357, U32 iStart)
{
    ASSERT(iStart & 1);

    for (U32 i = 0; i < kSieveUnroll * kSieveSize; i += kSieveSize)
        CopyBits(pSieve, i, pPrecomp357, 0, kSieveSize);

    U32 iPrime = 4;
    U32 P = 11;
    do
    {
        U32 j;
        const U32 Psqr = P * P;
        if (iStart > Psqr)
        {
            j = ((iStart - Psqr) >> 1) % P;
            if (j)
                j = P - j;
        }
        else
        {
            j = (Psqr - iStart) >> 1;
            if (j >= kSieveUnroll * kSieveSize)
                break;
        }

        for (; j < kSieveUnroll * kSieveSize; j += P)
            pSieve[j >> 6] &= ~(1ULL << (j & 63));

        P = kSmallPrimes[++iPrime];
    } while (P < kSieveUnroll * kSieveSize);

    pSieve[kNumSieveBlocks - 1] &= BuildBitMask(0, (kSieveUnroll * kSieveSize - 1) & 63);
}

static constexpr U64 Precomp357(U32 a, U32 b)
{
    U64 ret = 0;
    for (U32 i = 0; a < b; i += 2, a += 2)
    {
        if (a % 3 && a % 5 && a % 7)
            ret |= 1ULL << (i >> 1);
    }
    return ret;
}

static void EcmInternal(mpz_ptr ret, std::atomic_bool& found, std::atomic<U32>& curve, mpz_srcptr x, bool checkMtThresh)
{
    MONT_CREATE(O);
    MONT_CREATE(X);
    MONT_CREATE(Z);
    MONT_CREATE(S1);
    MONT_CREATE(S2);
    MONT_CREATE(S3);
    MONT_CREATE(S4);
    MONT_CREATE(S5);
    MONT_CREATE(S6);
    MONT_CREATE(S7);
    MONT_CREATE(S8);
    MONT_CREATE(S9);
    MONT_CREATE(S10);
    MONT_CREATE(S11);
    MONT_CREATE(S12);
    MONT_CREATE(origX);
    MONT_CREATE(origZ);
    MONT_CREATE(sigma);
    MONT_ARRAY_CREATE(root, kNumRoots);

    const MontgomerySystem ms(x);

    for (U32 iCurve = 0;;)
    {
        if (found || (checkMtThresh && iCurve >= kMtThresh - 1))
        {
            mpz_set_ui(ret, 0);
            return;
        }

        iCurve = curve.fetch_add(1);

        if (Lehmann(ret, x, iCurve))
            return;

#define MAKE_PRECOMP(p)                                   \
        {                                                 \
            Precomp357(p + 0 * 128, p + 1 * 128),         \
            Precomp357(p + 1 * 128, p + 2 * 128),         \
            Precomp357(p + 2 * 128, p + 3 * 128),         \
            Precomp357(p + 3 * 128, p + 2 * kSieveSize)   \
        }

        static constexpr U64 precomp357_1[] = MAKE_PRECOMP(1);
        static constexpr U64 precomp357_49[] = MAKE_PRECOMP(49);
        static constexpr U64 precomp357_229[] = MAKE_PRECOMP(229);
        static constexpr U64 precomp357_1011[] = MAKE_PRECOMP(1011);
        static constexpr U64 precomp357_3321[] = MAKE_PRECOMP(3321);

#undef MAKE_PRECOMP

        const U64* precomp357_P;

        U32 B1, B2, BP;
        if (iCurve < 26)
        {
            B1 = 2000;
            B2 = 200000;
            BP = 45;
            precomp357_P = precomp357_49;
        }
        else if (iCurve < 326)
        {
            B1 = 50000;
            B2 = 5000000;
            BP = 224;
            precomp357_P = precomp357_229;
        }
        else if (iCurve < 2000)
        {
            B1 = 1000000;
            B2 = 100000000;
            BP = 1001;
            precomp357_P = precomp357_1011;
        }
        else
        {
            B1 = 11000000;
            B2 = 1100000000;
            BP = 3316;
            precomp357_P = precomp357_3321;
        }

        {
            mpz_set_ui(S1.m_x, (U64)iCurve + 1);
            mpz_mul(S2.m_x, S1.m_x, S1.m_x);
            mpz_mul_ui(S2.m_x, S2.m_x, 3);
            mpz_sub_ui(S2.m_x, S2.m_x, 1);
            InvModChecked(S2.m_x, S2.m_x, x);
            mpz_add(S1.m_x, S1.m_x, S1.m_x);
            mpz_mul(S1.m_x, S1.m_x, S2.m_x);
            mpz_mod(S1.m_x, S1.m_x, x);
            mpz_mul(S2.m_x, S1.m_x, S1.m_x);
            mpz_mod(S2.m_x, S2.m_x, x);
            mpz_sub_ui(S3.m_x, S2.m_x, 1);
            mpz_mul(S3.m_x, S3.m_x, S1.m_x);
            mpz_mul_ui(S4.m_x, S2.m_x, 9);
            mpz_sub_ui(S4.m_x, S4.m_x, 1);
            mpz_mul(S3.m_x, S3.m_x, S4.m_x);
            if (mpz_divisible_p(S3.m_x, x))
                continue;
            mpz_mul(S4.m_x, S2.m_x, S2.m_x);
            mpz_mod(S4.m_x, S4.m_x, x);
            mpz_mul_si(S4.m_x, S4.m_x, -3);
            mpz_mul_ui(S3.m_x, S2.m_x, 6);
            mpz_sub(S4.m_x, S4.m_x, S3.m_x);
            mpz_add_ui(S4.m_x, S4.m_x, 1);
            mpz_mul(S3.m_x, S2.m_x, S1.m_x);
            mpz_mod(S3.m_x, S3.m_x, x);
            mpz_mul_2exp(S3.m_x, S3.m_x, 2);
            InvModChecked(S3.m_x, S3.m_x, x);
            mpz_mul(S4.m_x, S4.m_x, S3.m_x);
            mpz_mod(S4.m_x, S4.m_x, x);
            mpz_add_ui(S4.m_x, S4.m_x, 2);
            mpz_set_ui(S3.m_x, 4);
            InvModChecked(S3.m_x, S3.m_x, x);
            mpz_mul(S4.m_x, S4.m_x, S3.m_x);
            ms.ToMontgomery(O, S4.m_x);
            mpz_mul_ui(S2.m_x, S2.m_x, 3);
            mpz_add_ui(S2.m_x, S2.m_x, 1);
            ms.ToMontgomery(X, S2.m_x);
            mpz_mul_2exp(S1.m_x, S1.m_x, 2);
            ms.ToMontgomery(Z, S1.m_x);
        }

        U64 sieve[kNumSieveBlocks];

        // step 1 of 2
        {
            origX.DeepCopyFrom(X);
            origZ.DeepCopyFrom(Z);
            ms.MakeOne(sigma);

            for (U32 iIter = 0; iIter < 2; ++iIter)
            {
                for (U32 b = 1; b <= B1; b <<= 1)
                {
                    EcMul(ms, X, Z, S1, S2, X, Z, O);
                }
                for (U32 b = 3; b <= B1; b *= 3)
                {
                    EcMul(ms, S1, S2, S3, S4, X, Z, O);
                    EcAdd(ms, X, Z, S3, S4, S5, S6, X, Z, S1, S2, X, Z);
                }

                if (iIter == 0)
                {
                    ms.Mul(sigma, sigma, Z);
                }
                else
                {
                    ms.GcdN(ret, Z);
                    if (mpz_cmp_ui(ret, 1) > 0 && mpz_cmp(ret, x) != 0)
                        return;
                }

                U32 P;
                {
                    U32 iSieve = 1;
                    do
                    {
                        P = kSmallPrimes[iSieve++];
                        for (U32 b = P; b <= B1; b *= P)
                        {
                            EcPrac(ms, P, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, X, Z, O);
                        }
                        if (iIter == 0)
                        {
                            ms.Mul(sigma, sigma, Z);
                        }
                        else
                        {
                            ms.GcdN(ret, Z);
                            if (mpz_cmp_ui(ret, 1) > 0 && mpz_cmp(ret, x) != 0)
                                return;
                        }
                    } while (P <= BP);
                }

                P += 2;

                do
                {
                    MakeSieve(sieve, precomp357_P, P);

                    for (U32 iBlock = 0; iBlock < kNumSieveBlocks; ++iBlock)
                    {
                        U64 block = sieve[iBlock];
                        while (block)
                        {
                            const U32 i = (iBlock << 6) + CountTrailingZeros(block);
                            const U32 s = P + (i << 1);
                            if (s > B1)
                                goto label_exit_sieve;
                            EcPrac(ms, s, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, X, Z, O);
                            if (iIter == 0)
                            {
                                ms.Mul(sigma, sigma, Z);
                            }
                            else
                            {
                                ms.GcdN(ret, Z);
                                if (mpz_cmp_ui(ret, 1) > 0 && mpz_cmp(ret, x) != 0)
                                    return;
                            }

                            block &= block - 1;
                        }
                    }

                    label_exit_sieve:;

                    P += 2 * kSieveUnroll * kSieveSize;
                } while (P < B1);

                if (iIter == 0)
                {
                    if (ms.IsZero(sigma))
                    {
                        // retry on next iter, checking intermediates
                        X.DeepCopyFrom(origX);
                        Z.DeepCopyFrom(origZ);
                    }
                    else
                    {
                        // either step 1 has factored the number...

                        ms.GcdN(ret, sigma);
                        if (mpz_cmp_ui(ret, 1) > 0)
                            return;

                        // ...or it cannot, so break
                        break;
                    }
                }
            }
        }

        // step 2 of 2
        {
            origX.DeepCopyFrom(X);
            origZ.DeepCopyFrom(Z);
            for (U32 iIter = 0; iIter < 2; ++iIter)
            {
                ms.MakeOne(sigma);
                S8.DeepCopyFrom(X);
                S9.DeepCopyFrom(Z);
                if (!ms.InvMod(S3, Z))
                {
                    if (ms.IsZero(Z))
                        goto label_next_curve;

                    ms.GcdN(ret, Z);
                    ASSERT(mpz_cmp_ui(ret, 1) > 0 && mpz_cmp(ret, x) != 0);
                    return;
                }

                ms.Mul(root[0], X, S3);
                ms.Add(S1, X, Z);
                ms.Sqr(S1, S1);
                ms.Sub(S2, X, Z);
                ms.Sqr(S2, S2);
                ms.Mul(S10, S1, S2);
                ms.Sub(S1, S1, S2);
                EcMam(ms, S11, S1, O, S2);
                ms.Sub(S1, X, Z);
                ms.Add(S2, S10, S11);
                ms.Mul(S1, S1, S2);
                ms.Add(S3, X, Z);
                ms.Sub(S2, S10, S11);
                ms.Mul(S2, S2, S3);
                ms.Add(X, S1, S2);
                ms.Sqr(X, X);
                ms.Mul(X, X, S9);
                ms.Sub(Z, S1, S2);
                ms.Sqr(Z, Z);
                ms.Mul(Z, Z, S8);
                for (U32 iSieve = 5, iRoot = 0; iSieve < kSieveSize; iSieve += 2)
                {
                    S6.DeepCopyFrom(X);
                    S7.DeepCopyFrom(Z);
                    ms.Sub(S1, X, Z);
                    ms.Add(S3, S10, S11);
                    ms.Mul(S1, S1, S3);
                    ms.Add(S2, X, Z);
                    ms.Sub(S3, S10, S11);
                    ms.Mul(S2, S2, S3);
                    ms.Add(S3, S1, S2);
                    ms.Sqr(X, S3);
                    ms.Mul(X, X, S9);
                    ms.Sub(S3, S1, S2);
                    ms.Sqr(Z, S3);
                    ms.Mul(Z, Z, S8);
                    if (iIter == 0)
                    {
                        ms.Mul(sigma, sigma, S3);
                    }
                    else
                    {
                        ms.GcdN(ret, S3);
                        if (mpz_cmp_ui(ret, 1) > 0 && mpz_cmp(ret, x) != 0)
                            return;
                    }
                    if (iSieve == (kSieveSize / 2))
                    {
                        S4.DeepCopyFrom(X);
                        S5.DeepCopyFrom(Z);
                    }
                    if (iSieve % 3 && iSieve % 5 && iSieve % 7)
                    {
                        if (!ms.InvMod(S3, Z))
                        {
                            if (ms.IsZero(Z))
                                goto label_next_curve;

                            ms.GcdN(ret, Z);
                            ASSERT(mpz_cmp_ui(ret, 1) > 0 && mpz_cmp(ret, x) != 0);
                            return;
                        }

                        ms.Mul(root[++iRoot], X, S3);
                    }
                    S8.DeepCopyFrom(S6);
                    S9.DeepCopyFrom(S7);
                }

                ms.Add(S1, S4, S5);
                ms.Sqr(S1, S1);
                ms.Sub(S2, S4, S5);
                ms.Sqr(S2, S2);
                ms.Mul(X, S1, S2);
                ms.Sub(S1, S1, S2);
                EcMam(ms, Z, S1, O, S2);
                S8.DeepCopyFrom(X);
                S9.DeepCopyFrom(Z);
                ms.Add(S1, X, Z);
                ms.Sqr(S1, S1);
                ms.Sub(S2, X, Z);
                ms.Sqr(S2, S2);
                ms.Mul(S10, S1, S2);
                ms.Sub(S1, S1, S2);
                EcMam(ms, S11, S1, O, S2);

                const U32 iSieveMin = (B1 / (2 * kSieveSize));
                const U32 iSieveMax = (B2 / (2 * kSieveSize));
                for (U32 iSieve = 0; iSieve <= iSieveMax; ++iSieve)
                {
                    if (iSieve >= iSieveMin)
                    {
                        if (!ms.InvMod(S3, Z))
                        {
                            if (ms.IsZero(Z))
                                goto label_next_curve;

                            ms.GcdN(ret, Z);
                            ASSERT(mpz_cmp_ui(ret, 1) > 0 && mpz_cmp(ret, x) != 0);
                            return;
                        }

                        ms.Mul(S3, X, S3);

                        if (!(iSieve % kSieveUnroll) || iSieve == iSieveMin)
                            MakeSieve(sieve, precomp357_1, (iSieve / kSieveUnroll) * (2 * kSieveUnroll * kSieveSize) + 1);

                        {
                            U32 iSieveBlock = 0;
                            U64 block = precomp357_1[0];

                            for (U32 iRoot = 0, iSieveElem = (kSieveSize / 2) + (iSieve % kSieveUnroll) * kSieveSize; iRoot < kNumRoots; ++iRoot)
                            {
                                if (!block)
                                    block = precomp357_1[++iSieveBlock];
                                const U32 m = (iSieveBlock << 6) + CountTrailingZeros(block);
                                block &= block - 1;

                                const U32 k1 = iSieveElem + m;
                                const U32 k2 = iSieveElem - 1 - m;

                                if ((sieve[k1 >> 6] & (1ULL << (k1 & 63))) || (sieve[k2 >> 6] & (1ULL << (k2 & 63))))
                                {
                                    ms.Sub(S1, S3, root[iRoot]);
                                    ms.Mul(sigma, sigma, S1);
                                }
                            }
                        }

                        if (iIter == 1)
                        {
                            if (ms.IsZero(sigma))
                                goto label_next_curve;

                            ms.GcdN(ret, sigma);
                            if (mpz_cmp_ui(ret, 1) > 0)
                                return;
                        }
                    }

                    S6.DeepCopyFrom(X);
                    S7.DeepCopyFrom(Z);
                    ms.Sub(S1, X, Z);
                    ms.Add(S3, S10, S11);
                    ms.Mul(S1, S1, S3);
                    ms.Add(S2, X, Z);
                    ms.Sub(S3, S10, S11);
                    ms.Mul(S2, S2, S3);
                    ms.Add(X, S1, S2);
                    ms.Sqr(X, X);
                    ms.Mul(X, X, S9);
                    ms.Sub(Z, S1, S2);
                    ms.Sqr(Z, Z);
                    ms.Mul(Z, Z, S8);
                    S8.DeepCopyFrom(S6);
                    S9.DeepCopyFrom(S7);
                }

                if (iIter == 0)
                {
                    if (ms.IsZero(sigma))
                    {
                        // retry on next iter, checking intermediates
                        X.DeepCopyFrom(origX);
                        Z.DeepCopyFrom(origZ);
                    }
                    else
                    {
                        // either step 2 has factored the number...
                        ms.GcdN(ret, sigma);
                        if (mpz_cmp_ui(ret, 1) > 0)
                            return;

                        // ...or it cannot, so break
                        break;
                    }
                }
            }
        }
        // end step 2 of 2

        // on to next iCurve...
        label_next_curve:;
    }
}

static void EcmWorker(mpz_ptr ret, std::atomic_bool& found, std::atomic<U32>& curve, mpz_srcptr x, bool checkMtThresh)
{
    EcmInternal(ret, found, curve, x, checkMtThresh);
    if (mpz_cmp_ui(ret, 0) != 0)
    {
        found = true;
        ASSERT(mpz_cmp_ui(ret, 1) != 0 && mpz_cmp(ret, x) != 0);
    }
}

static U64 PerfectPower(mpz_ptr x)
{
    MPZ_CREATE(b);
    MPZ_CREATE(p);

    ASSERT(mpz_cmp_ui(x, 1) > 0);

    if (!mpz_tstbit(x, 0) && mpz_popcount(x) == 1)
    {
        const U64 e = mpz_sizeinbase(x, 2) - 1;
        mpz_set_ui(x, 2);
        return e;
    }

    U64 ret = 1;

label_iter:;

    constexpr double kLog2_3 = 1.5849625007211561;
    const double log2 = Log2(x);
    const U32 eMax = U32(::ceil(log2 / kLog2_3));
    for (U32 e = 2; e <= eMax; ++e)
    {
        const double l2e = log2 / e;
        if (l2e < 46.0)
        {
            const U64 ib = (U64)(I64)::round(::exp2(l2e));
            mpz_ui_pow_ui(p, ib, e);
            if (mpz_cmp(p, x) == 0)
            {
                ret *= e;
                mpz_set_ui(x, ib);
                goto label_iter;
            }
        }
        else
        {
            mpz_rootrem(b, p, x, e);
            if (mpz_cmp_ui(p, 0) == 0)
            {
                ret *= e;
                mpz_set(x, b);
                goto label_iter;
            }
        }
    }

    return ret;
}

static void Ecm(std::vector<FactorInfo>& v, U32 numThreads, std::atomic<U32>& curve, mpz_srcptr x)
{
    constexpr U32 kMaxThreads = 32;

    mpz_t res[kMaxThreads];

    if (numThreads == 0)
        numThreads = std::thread::hardware_concurrency();
    if (numThreads > kMaxThreads)
        numThreads = kMaxThreads;

    std::atomic_bool found = false;

    if (numThreads > 1)
    {
        if (curve < kMtThresh)
        {
            mpz_init(res[0]);
            EcmWorker(res[0], found, curve, x, true);
            if (found)
                RecordFactor(v, res[0]);
            mpz_clear(res[0]);
            if (found)
                return;
        }

        std::thread threads[kMaxThreads];
        for (U32 i = 0; i < numThreads; ++i)
            mpz_init(res[i]);
        for (U32 i = 0; i < numThreads; ++i)
            threads[i] = std::thread(EcmWorker, res[i], std::ref(found), std::ref(curve), x, false);
        for (U32 i = 0; i < numThreads; ++i)
            threads[i].join();
        for (U32 i = 0; i < numThreads; ++i)
            if (mpz_cmp_ui(res[i], 0) != 0)
                RecordFactor(v, res[i]);
        for (U32 i = 0; i < numThreads; ++i)
            mpz_clear(res[i]);
    }
    else
    {
        mpz_init(res[0]);
        EcmWorker(res[0], found, curve, x, false);
        if (found)
            RecordFactor(v, res[0]);
        mpz_clear(res[0]);
    }
}

static bool PM1(std::vector<FactorInfo>& v, mpz_srcptr x, mpz_srcptr b, U64 e, I32 ofs)
{
    MPZ_CREATE(t);

    bool updated = false;

    if (ofs == -1)
    {
        for (U64 f = e; !(f & 1);)
        {
            f >>= 1;
            mpz_pow_ui(t, b, f);
            mpz_sub_ui(t, t, 1);
            if (RecordFactor(v, t))
                updated = true;
            if (Aurifeuille(v, b, f, 1))
                updated = true;
        }
    }

    for (U64 k = 1; k * k <= e; ++k)
    {
        if (e % k == 0)
        {
            for (const U64 f : {e / k, k})
            {
                if ((e / f) & 1)
                {
                    mpz_pow_ui(t, b, f);
                    mpz_add_si(t, t, ofs);
                    mpz_gcd(t, t, x);
                    if (RecordFactor(v, t))
                        updated = true;
                    ASSERT(mpz_divisible_p(x, t));
                    mpz_divexact(t, x, t);
                    if (RecordFactor(v, t))
                        updated = true;
                    if (Aurifeuille(v, b, f, ofs))
                        updated = true;
                }
            }
        }
    }

    return updated;
}

std::vector<FactorInfo> Factorize(mpz_srcptr n, U32 numThreads /*= 0*/)
{
    MPZ_CREATE(x);
    MPZ_CREATE(b);
    MPZ_CREATE(c);

    ASSERT(mpz_cmp_ui(n, 0) > 0);

    std::vector<FactorInfo> ret;
    if (mpz_cmp_ui(n, 1) <= 0)
        return ret;

    mpz_set(x, n);

    // powers of 2
    {
        const U64 e = CountTrailingZeros(x);
        if (e)
        {
            ret.emplace_back(2, e);
            mpz_div_2exp(x, x, e);
        }
    }

    // trial division
    if (TrialDiv(ret, x))
    {
        if (mpz_cmp_ui(x, 1) != 0)
            ret.emplace_back(x, 1);
        return ret;
    }

    std::vector<FactorInfo> v;
    v.emplace_back(x, 1);

    std::atomic<U32> curve = 1;

label_loop_restart:;
    for (U64 i = 0; i < v.size(); ++i)
    {
        FactorInfo& info = v[i];

        if (mpz_cmp_ui(info.m_factor, kMaxTrialDivSqr) <= 0)
        {
            RecordPrime(ret, v, i--);
            continue;
        }

        // perfect power
        {
            const U64 e = PerfectPower(info.m_factor);
            if (e != 1)
            {
                info.m_exp *= e;

                if (mpz_cmp_ui(info.m_factor, kMaxTrialDivSqr) <= 0)
                {
                    RecordPrime(ret, v, i--);
                    continue;
                }
            }
        }

        // perfect power +/- 1
        for (const I32 ofs : {-1, 1})
        {
            mpz_add_si(b, info.m_factor, -ofs);
            const U64 e = PerfectPower(b);
            if (e != 1)
            {
                mpz_set(c, info.m_factor);
                if (PM1(v, c, b, e, ofs))
                    goto label_loop_restart;
            }
        }

        // primality test
        const BpswResult primality = BpswPrimalityTest(info.m_factor);
        if (primality == BpswResult::kProbablePrime)
        {
            RecordPrime(ret, v, i--);
            continue;
        }

        // Fermat pseudoprimes
        if (primality == BpswResult::kCompositeFermatPseudoprime)
        {
            mpz_set(c, info.m_factor);
            if (Fermat(v, c))
                goto label_loop_restart;
        }

        // Lehmann/ECM
        {
            mpz_set(c, info.m_factor);
            Ecm(v, numThreads, curve, c);
            goto label_loop_restart;
        }
    }

    ASSERT(v.empty());
    Sort(ret);
    return ret;
}
