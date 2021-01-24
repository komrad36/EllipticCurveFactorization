# EllipticCurveFactorization
Fast, single-file, MIT-licensed large integer factorization using ECM combined with other techniques.

NO dependencies at all except GNU MP (a.k.a GMP, a.k.a. MPIR on Windows), the standard BigInteger library.

Linux/Windows both supported - libgmp on Linux, mpir.lib on Windows.

Simple call:

`std::vector<FactorInfo> Factorize(mpz_srcptr n);`

You are returned a vector of `FactorInfo` structs, each containing an `mpz_t` (the factor) and a `uint64` (the exponent).

Returning from the function transfers ownership of the `mpz_t`s within. As the caller, YOU are responsible for `mpz_clear`ing them when you're done with them.

Multi-threaded by default, with automatic selection of thread count. You can override with a custom thread count, including disabling MT by requesting just 1 thread, using a second argument:

`std::vector<FactorInfo> Factorize(mpz_srcptr n, uint32_t numThreads = 0);`

0 is the default and means automatic selection of thread count.
