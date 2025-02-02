#include <cassert>
#include <iostream>

#define GLUE_DEBUG(...) printf(__VA_ARGS__)
#include "glue.hpp"

// g++ -std=c++17 -o test_glue test_glue.cpp && ./test_glue

bool cmp_float(float a, float b) {
  return std::abs(a - b) < 1e-6;
}

static glue_outbuf outbuf;

void test_load_req() {
  glue_msg_load_req req;
  req.use_mmap.value = true;
  req.n_gpu_layers.value = 32;
  req.seed.value = 42;
  req.n_ctx.value = 2048;
  req.embeddings.value = false;
  req.pooling_type.value = "mean";

  req.handler.serialize(outbuf);
  FILE* fp = fopen("dump.bin", "wb");
  fwrite(outbuf.data.data(), 1, outbuf.data.size(), fp);
  fclose(fp);

  printf("\n----------\n\n");
  
  glue_msg_load_req req2;
  glue_inbuf inbuf(outbuf.data.data());
  req2.handler.deserialize(inbuf);

  assert(req2.use_mmap.value == true);
  assert(req2.n_gpu_layers.value == 32); 
  assert(req2.seed.value == 42);
  assert(req2.n_ctx.value == 2048);
  assert(req2.embeddings.value == false);
  assert(req2.pooling_type.value == "mean");
}

void test_sampling_init() {
  glue_msg_sampling_init_req req;
  req.mirostat.value = 2;
  req.temp.value = 0.8;
  req.top_p.value = 0.95;
  req.penalty_repeat.value = 1.1;
  req.grammar.value = "test grammar";
  std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
  req.tokens.arr = tokens;

  req.handler.serialize(outbuf);
  FILE* fp = fopen("dump2.bin", "wb");
  fwrite(outbuf.data.data(), 1, outbuf.data.size(), fp);
  fclose(fp);

  printf("\n----------\n\n");

  glue_msg_sampling_init_req req2;
  glue_inbuf inbuf(outbuf.data.data());
  req2.handler.deserialize(inbuf);

  assert(req2.mirostat.value == 2);
  assert(cmp_float(req2.temp.value, 0.8));
  assert(cmp_float(req2.top_p.value, 0.95)); 
  assert(cmp_float(req2.penalty_repeat.value, 1.1));
  assert(req2.grammar.value == "test grammar");
  for (int i = 0; i < tokens.size(); i++) {
    assert(req2.tokens.arr[i] == tokens[i]);
  }
}

int main() {
  test_load_req();
  printf("\n\n\n\n");
  test_sampling_init();
  std::cout << "All tests passed!" << std::endl;
  return 0;
}
