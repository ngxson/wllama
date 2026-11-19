#pragma once

#include <stdint.h>

// on wasm64, int64_t is long long but INT64_C gives long
#undef INT64_C
#define INT64_C(c) c##LL
