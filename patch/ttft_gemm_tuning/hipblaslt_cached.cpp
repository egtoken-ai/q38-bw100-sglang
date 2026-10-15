// SPDX-License-Identifier: Apache-2.0
// Experimental DTK hipBLASLt W8A8 channelwise C ABI. No Torch C++ dependency.
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
thread_local std::string last_error;
std::mutex lock;
void check(hipblasStatus_t s, const char* where) {
  if (s != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(where)+" status="+std::to_string(s));
}
void hipcheck(hipError_t s, const char* where) {
  if (s != hipSuccess) throw std::runtime_error(std::string(where)+": "+hipGetErrorString(s));
}
struct Problem {
  hipblasLtMatmulDesc_t desc=nullptr;
  hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,d=nullptr;
  hipblasLtMatmulAlgo_t algo{};
  size_t required=0;
  int index=-1;
  ~Problem() {
    if(a) hipblasLtMatrixLayoutDestroy(a); if(b) hipblasLtMatrixLayoutDestroy(b);
    if(c) hipblasLtMatrixLayoutDestroy(c); if(d) hipblasLtMatrixLayoutDestroy(d);
    if(desc) hipblasLtMatmulDescDestroy(desc);
  }
};
struct StreamCache {
  hipblasLtHandle_t handle=nullptr;
  void* workspace=nullptr;
  size_t workspace_size=0;
  // beta is always zero. Allocate a correctly sized INT32 C buffer anyway,
  // rather than aliasing the smaller BF16 D buffer as the legacy wrapper does.
  void* c_buffer=nullptr;
  size_t c_size=0;
  std::map<std::tuple<int64_t,int64_t,int64_t,int,size_t>,std::unique_ptr<Problem>> problems;
  StreamCache() { check(hipblasLtCreate(&handle), "create handle"); }
  ~StreamCache() {
    problems.clear(); if(workspace) hipFree(workspace); if(c_buffer) hipFree(c_buffer);
    if(handle) hipblasLtDestroy(handle);
  }
};
// Leaked registry avoids GPU teardown ordering at interpreter exit. Process exit
// releases resources. q38_lt_clear explicitly synchronizes and frees a device.
auto& caches() {
  static auto* p = new std::map<std::pair<int,uintptr_t>,std::unique_ptr<StreamCache>>;
  return *p;
}
void grow(void*& ptr, size_t& capacity, size_t bytes) {
  if(bytes<=capacity) return;
  void* replacement=nullptr;
  hipcheck(hipMalloc(&replacement, bytes), "allocate cached buffer");
  if(ptr) hipcheck(hipFree(ptr), "free old cached buffer");
  ptr=replacement; capacity=bytes;
}
void scales(Problem& p, const void* weight_scale, const void* activation_scale) {
  check(hipblasLtMatmulDescSetAttribute(p.desc,HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                      &weight_scale,sizeof(weight_scale)), "A weight scale");
  check(hipblasLtMatmulDescSetAttribute(p.desc,HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                      &activation_scale,sizeof(activation_scale)), "B activation scale");
}
std::unique_ptr<Problem> prepare(StreamCache& cache, int64_t m,int64_t n,int64_t k,
                                int index,size_t limit,const void* ws,const void* as) {
  auto p=std::make_unique<Problem>();
  check(hipblasLtMatmulDescCreate(&p->desc,HIPBLAS_COMPUTE_32I,HIP_R_32F), "descriptor");
  hipblasOperation_t transa=HIPBLAS_OP_T, transb=HIPBLAS_OP_N;
  check(hipblasLtMatmulDescSetAttribute(p->desc,HIPBLASLT_MATMUL_DESC_TRANSA,&transa,sizeof(transa)),"transA");
  check(hipblasLtMatmulDescSetAttribute(p->desc,HIPBLASLT_MATMUL_DESC_TRANSB,&transb,sizeof(transb)),"transB");
  hipblasLtMatmulMatrixScale_t mode=HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  check(hipblasLtMatmulDescSetAttribute(p->desc,HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)),"A vector mode");
  check(hipblasLtMatmulDescSetAttribute(p->desc,HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)),"B vector mode");
  // PyTorch A=[M,K], W=[N,K] row-major. BLAS computes W^T-like view @ A
  // in column-major: stored A=[K,N], B=[K,M], C/D=[N,M].
  check(hipblasLtMatrixLayoutCreate(&p->a,HIP_R_8I,k,n,k),"A layout");
  check(hipblasLtMatrixLayoutCreate(&p->b,HIP_R_8I,k,m,k),"B layout");
  check(hipblasLtMatrixLayoutCreate(&p->c,HIP_R_32I,n,m,n),"C layout");
  check(hipblasLtMatrixLayoutCreate(&p->d,HIP_R_16BF,n,m,n),"D layout");
  scales(*p,ws,as);
  if(index>=0) {
    std::vector<int> ids{index};
    std::vector<hipblasLtMatmulHeuristicResult_t> selected;
    check(hipblaslt_ext::getAlgosFromIndex(cache.handle,ids,selected),"get algorithm index");
    if(selected.size()!=1) throw std::runtime_error("explicit algorithm index did not resolve uniquely");
    p->algo=selected[0].algo;
  } else {
    hipblasLtMatmulPreference_t pref=nullptr;
    check(hipblasLtMatmulPreferenceCreate(&pref),"preference");
    auto status=hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit));
    if(status!=HIPBLAS_STATUS_SUCCESS) { hipblasLtMatmulPreferenceDestroy(pref); check(status,"workspace preference"); }
    hipblasLtMatmulHeuristicResult_t result{}; int count=0;
    status=hipblasLtMatmulAlgoGetHeuristic(cache.handle,p->desc,p->a,p->b,p->c,p->d,pref,1,&result,&count);
    hipblasLtMatmulPreferenceDestroy(pref);
    check(status,"heuristic");
    if(count!=1 || result.state!=HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("no supported heuristic");
    p->algo=result.algo;
  }
  // DTK COMPUTE_32I consumes integer alpha/beta, matching the native
  // torch.int32 CPU tensors. FP32 vector scales and descriptor scale type
  // do not make these scalar pointers float (1.0f bits would be 1065353216).
  int32_t alpha=1,beta=0;
  check(hipblaslt_ext::matmulIsAlgoSupported(cache.handle,p->desc,&alpha,p->a,p->b,&beta,p->c,p->d,p->algo,p->required),"algorithm supports exact problem");
  if(p->required>limit) throw std::runtime_error("algorithm requires more workspace than budget");
  p->index=hipblaslt_ext::getIndexFromAlgo(p->algo);
  if(index>=0 && p->index!=index) throw std::runtime_error("resolved algorithm index differs");
  return p;
}
}
extern "C" const char* q38_lt_last_error() { return last_error.c_str(); }
extern "C" int q38_lt_run(int device,uintptr_t stream,const void* activation,const void* weight,
                           const void* activation_scale,const void* weight_scale,void* output,
                           int64_t m,int64_t n,int64_t k,int index,size_t workspace_limit,
                           int* chosen_index,size_t* required_workspace) {
  try {
    std::lock_guard<std::mutex> guard(lock);
    if(!activation||!weight||!activation_scale||!weight_scale||!output||m<=0||n<=0||k<=0)
      throw std::runtime_error("null pointer or invalid shape");
    if(workspace_limit>size_t(128)*1024*1024) throw std::runtime_error("workspace budget exceeds 128MiB");
    hipcheck(hipSetDevice(device),"set device");
    auto& ptr=caches()[{device,stream}];
    if(!ptr) ptr=std::make_unique<StreamCache>();
    auto& cache=*ptr;
    auto key=std::make_tuple(m,n,k,index,workspace_limit);
    auto found=cache.problems.find(key);
    if(found==cache.problems.end()) {
      auto problem=prepare(cache,m,n,k,index,workspace_limit,weight_scale,activation_scale);
      found=cache.problems.emplace(key,std::move(problem)).first;
    }
    auto& p=*found->second;
    grow(cache.workspace,cache.workspace_size,p.required);
    grow(cache.c_buffer,cache.c_size,size_t(m)*size_t(n)*sizeof(int32_t));
    scales(p,weight_scale,activation_scale);
    // DTK COMPUTE_32I consumes integer alpha/beta, matching the native
  // torch.int32 CPU tensors. FP32 vector scales and descriptor scale type
  // do not make these scalar pointers float (1.0f bits would be 1065353216).
  int32_t alpha=1,beta=0;
    check(hipblasLtMatmul(cache.handle,p.desc,&alpha,weight,p.a,activation,p.b,
                         &beta,cache.c_buffer,p.c,output,p.d,&p.algo,
                         cache.workspace,p.required,reinterpret_cast<hipStream_t>(stream)),"matmul");
    if(chosen_index) *chosen_index=p.index;
    if(required_workspace) *required_workspace=p.required;
    last_error.clear(); return 0;
  } catch(const std::exception& e) { last_error=e.what(); return 1; }
}
extern "C" int q38_lt_clear(int device) {
  try {
    std::lock_guard<std::mutex> guard(lock);
    hipcheck(hipSetDevice(device),"set device");
    hipcheck(hipDeviceSynchronize(),"synchronize before cache cleanup");
    for(auto it=caches().begin();it!=caches().end();) {
      if(it->first.first==device) it=caches().erase(it); else ++it;
    }
    return 0;
  } catch(const std::exception& e) { last_error=e.what(); return 1; }
}
