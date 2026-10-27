// soga_inputs.h: input generation for the SOGA validation campaigns.
// Portable PRNG (splitmix64), so A, B and C are bit-identical on every machine.

#ifndef SOGA_INPUTS_H
#define SOGA_INPUTS_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define SEED_BC 20261026ULL   /* B and C: same values in all four campaigns */
#define SEED_A  20261027ULL   /* A: one seed, the distribution changes      */

enum dist_a { DIST_U01 = 0, DIST_NORMAL = 1, DIST_SPARSE = 2, DIST_U010 = 3, DIST_INVALID = -1 };

static enum dist_a parse_dist(const char *s)
{
	if (strcmp(s, "u01") == 0)    return DIST_U01;     /* A ~ U[0,1]                       */
	if (strcmp(s, "normal") == 0) return DIST_NORMAL;  /* A ~ N(0, 1/3), variance 1/3      */
	if (strcmp(s, "sparse") == 0) return DIST_SPARSE;  /* A = 0 w.p. 0.9, else U[0,sqrt10] */
	if (strcmp(s, "u010") == 0)   return DIST_U010;    /* A ~ U[0,10]                      */
	return DIST_INVALID;
}

static uint64_t rng_state;

static void rng_seed(uint64_t s) { rng_state = s; }

static double u01(void)   /* uniform in [0,1), 53-bit resolution */
{
	uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	z ^= z >> 31;
	return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

static double sample_a(enum dist_a d)
{
	switch (d) {
	case DIST_U01:    return u01();
	case DIST_NORMAL: {   /* Box-Muller; 1 - u01() is in (0,1], so log() is finite */
		double r = sqrt(-2.0 * log(1.0 - u01()));
		return sqrt(1.0 / 3.0) * r * cos(2.0 * M_PI * u01());
	}
	case DIST_SPARSE: return (u01() < 0.1) ? sqrt(10.0) * u01() : 0.0;
	case DIST_U010:   return 10.0 * u01();
	default:          return NAN;
	}
}

#endif /* SOGA_INPUTS_H */
