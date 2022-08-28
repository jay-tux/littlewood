#include <iostream>
#include <cmath>
#include <chrono>

//complex operations
#define cadd(a, b) (vec2(a.x + b.x,              a.y + b.y))
#define csub(a, b) (vec2(a.x - b.x,              a.y - b.y))
#define cmul(a, b) (vec2(a.x * b.x - a.y * b.y,  a.x * b.y + a.y * b.x))
#define cinv(a)    (vec2(a.x / sqrt(norm(a)),   -a.y / sqrt(norm(a))))
#define norm(a)    (a.x * a.x + a.y * a.y)

#define prc 1e-10
#define sz 1024.0f
#define sqrt(a) std::sqrt(a)
#define prvc2(a) a.x << " + " << a.y << "i" << ", norm: " << sqrt(norm(a))
#define ABS(a) ((a) < 0 ? (-a) : (a))

struct vec2 {
  vec2() : x{0.0f}, y{0.0f} {}
  vec2(float x, float y) : x{x}, y{y} {}
  float x, y;
};

void isSolution(vec2 z, bool &res, int &it) {
  float tol = prc * prc;
  float nrm = norm(z);
  it = 0;
  if(ABS(nrm-1.0f) < prc) { res = true; return; }
  // |z| <= (1/2)  <=> norm(z) <= 1/4
  // |z| >= 2      <=> norm(z) >= 4
  if(nrm <= 0.25f || nrm >= 4.0f) { res = false; return; }
  // (1/sqrt[4](2)) < |z| < sqrt[4](2) <=> 1/sqrt(2) < norm(z) < sqrt(2)
  //                                   <=> 1/2 < norm(z) * norm(z) < 2
  //if(nrm * nrm > 0.5f && nrm * nrm < 2.0f) { res = true; return; }

  if(z.x < 0) z.x = -z.x;    //mirror to positive x (around y axis)
  if(z.y < 0) z.y = -z.y;    //mirror to positive y (around x axis)
  if(nrm > 1) z   = cinv(z); //mirror to |z| < 1    (around unit circle)

  vec2 curr, px, low, high, prev;
  prev = vec2(0.0f, 0.0f);
  curr = z;
  px   = vec2(1.0f, 0.0f);
  nrm  = nrm / (1 - 2 * sqrt(nrm) + nrm);
  //printf("--- Starting loop ---\n");

  while(it < 100000 && norm(csub(curr, prev)) > tol) {
    low  = csub(px, curr);
    high = cadd(px, curr);
    /*std::cout << "==ITERATION " << it << "==" << std::endl
              << " -> curr:  " << prvc2(curr) << std::endl
              << " -> low:   " << prvc2(low)  << std::endl
              << " -> high:  " << prvc2(high) << std::endl
              << " -> delta: " << norm(csub(curr, prev)) << std::endl << std::endl;*/
    if(norm(low) <= tol || norm(high) <= tol) { res = true; return; }
    if(norm(low) > nrm && norm(high) > nrm) { res = false; return; }

    prev = curr;
    curr = cmul(curr, z);
    nrm  = nrm * norm(z);
    px   = (norm(low) < norm(high)) ? low : high;
    it++;
  }
  res = false;
}

int main(int argc, char **argv) {
  bool sol;
  int it;
  vec2 pos = { 0.0f, 0.0f };
  for(int x = 0; x < sz; x++) {
    for(int y = 0; y < sz; y++) {
      pos.x = (float)x / (sz) - 2.0f; //(float)x / (sz/4) - 2.0f;
      pos.y = (float)y / (sz) - 0.5f; //(float)y / (sz/4) - 2.0f;
      float diff = norm(pos) - 1.0f;
      /*if((diff > 0 ? diff : -diff) <= prc)
        std::cout << prvc2(pos) << " has norm 1.0f +/- " << prc << "; diff: " << (norm(pos) - 1.0f) << std::endl;*/
      std::cout << pos.x << "," << pos.y << "," << std::flush;
      auto start = std::chrono::steady_clock::now();
      isSolution(pos, sol, it);
      auto end = std::chrono::steady_clock::now();
      //format: x,y,duration,solution,iterations
      std::cout << (end - start).count() << ","
                << (sol ? "true" : "false") << "," << it << std::endl;
    }
  }
  /*vec2 pos = { 0.0f, -1.0f };
  isSolution(pos, sol, it);*/
}
