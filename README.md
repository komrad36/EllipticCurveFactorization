# EllipticCurveFactorization
Fast, single-file, MIT-licensed large integer factorization using ECM combined with other techniques.

Absolutely demolishes Flint, GMP-ECM, and other ECM-based factorization libraries in performance (though if you need to factor HUGE numbers, better, more complicated techniques than ECM exist).

NO dependencies at all except GNU MP (a.k.a GMP, a.k.a. MPIR on Windows), the standard Big Integer library.

Linux/Windows both supported - `libgmp` on Linux, `mpir.lib` on Windows.

Be sure to compile GMP/MPIR configured specific to your architecture. The generic version is often more than `2.5x` slower.

Simple call:

```cpp
std::vector<FactorInfo> Factorize(mpz_srcptr n);
```

You are returned a vector of `FactorInfo` structs, each containing an `mpz_t` (the factor) and a `uint64_t` (the exponent).

Returning from the function transfers ownership of the `mpz_t`s within. As the caller, YOU are responsible for `mpz_clear`ing them when you're done with them.

Multi-threaded by default, with automatic selection of thread count. You can override with a custom thread count, including disabling MT by requesting just 1 thread, using a second argument:

```cpp
std::vector<FactorInfo> Factorize(mpz_srcptr n, uint32_t numThreads = 0);
```

`0` is the default and means automatic selection of thread count.

For machine-word-sized (32- or 64-bit) integers, see https://github.com/komrad36/Factorization-Primality.
