#include "kernel.h"

__device__ cuComplex operator+(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
__device__ cuComplex operator-(cuComplex a, cuComplex b) { return cuCsubf(a, b); }
__device__ cuComplex operator*(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
__device__ cuComplex operator/(cuComplex a, cuComplex b) { return cuCdivf(a, b); }
__device__ cuComplex operator+(cuComplex a, float     b) { return a + ccompf(b); }
__device__ cuComplex operator-(cuComplex a, float     b) { return a - ccompf(b); }
__device__ cuComplex operator*(cuComplex a, float     b) { return a * ccompf(b); }
__device__ cuComplex operator/(cuComplex a, float     b) { return a / ccompf(b); }
__device__ cuComplex operator+(float     a, cuComplex b) { return ccompf(a) + b; }
__device__ cuComplex operator-(float     a, cuComplex b) { return ccompf(a) - b; }
__device__ cuComplex operator*(float     a, cuComplex b) { return ccompf(a) * b; }
__device__ cuComplex operator/(float     a, cuComplex b) { return ccompf(a) / b; }
__device__ float     operator!(cuComplex a)              { return cuCabsf(a);    }
__device__ cuComplex operator~(cuComplex a)              { return 1.0f   /    a; }

__device__ bool isRoot(cuComplex z, float tol, int *it, int maxit) {
  //cuComplex z = ccomp(c.x, c.y);
  float nrm = !z;
  *it = 0;

  if(abs(nrm - 1.0f) < tol) { return true; }
  if(nrm <= 0.25f || nrm > 4.0f) { return false; }

  z = ccomp(abs(z.x), abs(z.y));
  if(nrm > 1.0f) z = ~z;

  cuComplex curr = z, px = { 1.0f, 0.0f }, low, high, prev = { 0.0f, 0.0f };
  nrm = !z / (1 - !z);

  while(*it < maxit && (!(curr - prev)) > tol) {
    low  = px - curr;
    high = px + curr;

    if((!low) <= tol || (!high) <= tol) { return true;  }
    if((!low) >  nrm && (!high) >  nrm) { return false; }

    prev = curr;
    curr = curr * z;
    nrm  = nrm * (!z);
    px   = ((!low) < (!high)) ? low : high;
    (*it)++;
  }

  return false;
}

__device__ cuComplex toComplex(complex &pix, complex &bounds, transform &t)
{
  return make_cuComplex(
    (4 * pix.x) / (bounds.x * t.zoom) - 2.0f / t.zoom + t.center.x,
    (4 * pix.y) / (bounds.y * t.zoom) - 2.0f / t.zoom + t.center.y
  );
}

__global__ void kernel(float *buf, float *itbuf, int w, int h, transform t, float tol, int maxit) {
  int x = threadIdx.x;
  int y = blockIdx.x;
  int it = 0;
  complex p = { (float)x, (float)y }, bounds = { (float)w, (float)h };
  cuComplex z = toComplex(p, bounds, t);
  int pix = (y * w + x);
  buf[pix] = isRoot(z, tol, &it, maxit);
  itbuf[pix] = log((float)(it)) / log((float)maxit);
}

void callKernel(cuData &data, cuGL &buf, transform &t) {
  kernel<<<data.height, data.width, 0>>>(buf.buffer, buf.itbuffer, data.width, data.height, t, data.tol, data.maxit);
  cudaDeviceSynchronize();
  updateTexture(data, buf);
}

bool transformer::changed() const {
  return up || down || left || right || in || out || reset;
}

void recenter(transformer &delta, transform &t)
{
  if(delta.in)        t.zoom *= 1.1f;
  if(delta.out)       t.zoom /= 1.1f;

  float mv = 1 /      t.zoom * 0.25f;
  if(delta.up)        t.center.y += mv;
  if(delta.down)      t.center.y -= mv;
  if(delta.right)     t.center.x += mv;
  if(delta.left)      t.center.x -= mv;

  if(delta.reset)     t = { { 0.0f, 0.0f }, 1.0f, true };

  if(delta.changed()) t.hasChanged = true;
}
