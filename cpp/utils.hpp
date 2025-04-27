#pragma once

#include <cstdint>
#include <stdio.h>

#include "llama.h"
#include "mtmd.h"

//
// utils
//

static uint64_t fnv_hash(const uint8_t *data, size_t len)
{
  const uint64_t fnv_prime = 0x100000001b3ULL;
  uint64_t hash = 0xcbf29ce484222325ULL;

  for (size_t i = 0; i < len; ++i)
  {
    hash ^= data[i];
    hash *= fnv_prime;
  }
  return hash;
}
