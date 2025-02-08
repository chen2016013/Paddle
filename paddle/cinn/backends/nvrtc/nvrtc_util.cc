// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include <fstream>
#include <iostream>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/nvrtc/header_generator.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
PD_DECLARE_string(cinn_nvcc_cmd_path);
PD_DECLARE_string(nvidia_package_dir);
PD_DECLARE_bool(nvrtc_compile_to_cubin);
PD_DECLARE_bool(cinn_nvrtc_cubin_with_fmad);

namespace cinn {
namespace backends {
namespace nvrtc {

static bool TryLocatePath(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static std::vector<std::string> GetNvidiaAllIncludePath(
    const std::string& nvidia_package_dir) {
  std::vector<std::string> include_paths;
  const std::string delimiter = "/";
  // Expand this list if necessary.
  const std::vector<std::string> sub_modules = {"cublas",
                                                "cudnn",
                                                "cufft",
                                                "cusparse",
                                                "cusolver",
                                                "cuda_nvrtc",
                                                "curand",
                                                "cuda_runtime"};
  for (auto& sub_module : sub_modules) {
    std::string path =
        nvidia_package_dir + delimiter + sub_module + delimiter + "include";
    include_paths.push_back(path);
  }
  return include_paths;
}

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  if (runtime::CanUseNvccCompiler()) {
    return CompileWithNvcc(code);
  }
  return CompileCudaSource(code, include_headers);
}

Compiler::Compiler() {
  if (FLAGS_nvrtc_compile_to_cubin) {
#if CUDA_VERSION >= 11010
    compile_to_cubin_ = true;
#endif
  }
  VLOG(4) << "FLAGS_nvrtc_compile_to_cubin: " << FLAGS_nvrtc_compile_to_cubin
          << ", compile_to_cubin_: " << compile_to_cubin_;
}

bool Compiler::compile_to_cubin() { return compile_to_cubin_; }

std::vector<std::string> Compiler::FindCUDAIncludePaths() {
  const std::string delimiter = "/";
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path += cuda_path_env;
    cuda_include_path += delimiter + "include";
    VLOG(4) << "FindCUDAIncludePaths from CUDA_PATH: " << cuda_include_path;
    return {cuda_include_path};
  }

#if defined(__linux__)
  if (!FLAGS_nvidia_package_dir.empty() &&
      TryLocatePath(FLAGS_nvidia_package_dir)) {
    VLOG(4) << "FindCUDAIncludePaths from nvidia_package_dir: "
            << FLAGS_nvidia_package_dir;
    return GetNvidiaAllIncludePath(FLAGS_nvidia_package_dir);
  }

  cuda_include_path = "/usr/local/cuda/include";
  if (TryLocatePath(cuda_include_path)) {
    VLOG(4) << "FindCUDAIncludePaths from " << cuda_include_path;
    return {cuda_include_path};
  }
#endif
  std::stringstream ss;
  ss << "Cannot find cuda include path."
     << "CUDA_PATH is not set or CUDA is not installed in the default "
        "installation path."
     << "In other than linux, it is necessary to set CUDA_PATH.";
  PADDLE_THROW(::common::errors::Fatal(ss.str()));
  return {cuda_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string processString(const std::string& s) {
  if (s.substr(0, 6) == "extern" || s.substr(0, 10) == "__global__") {
    std::vector<int> leftBraces;  // 存储所有左花括号 '{' 的位置
    for (int i = 0; i < s.size(); ++i) {
      if (s[i] == '{') {
        leftBraces.push_back(i);
      }
    }
    // 如果没有至少两个 '{'，直接返回原字符串
    if (leftBraces.size() < 2) {
      return s;
    }
    int start = leftBraces[1];  // 第二个 '{' 的位置

    std::vector<int> rightBraces;  // 存储所有右花括号 '}' 的位置
    for (int i = 0; i < s.size(); ++i) {
      if (s[i] == '}') {
        rightBraces.push_back(i);
      }
    }
    // 如果没有至少两个 '}'，直接返回原字符串
    if (rightBraces.size() < 2) {
      return s;
    }
    int end = rightBraces[rightBraces.size() - 2];  // 倒数第二个 '}' 的位置

    // 如果第二个 '{' 在倒数第二个 '}' 之后，不处理
    if (start >= end) {
      return s;
    }

    // 拼接删除区间后的字符串
    return s.substr(0, start + 1) + s.substr(end);
  } else {
    return s;
  }
}

std::string Compiler::CompileCudaSource(const std::string& code,
                                        bool include_headers) {
  const auto& header_gen = JitSafeHeaderGenerator::GetInstance();
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  cudaError_t e1 =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }
  if (compile_to_cubin_) {
    compile_options.push_back("-arch=sm_" + cc);
    std::string enable_fmad =
        FLAGS_cinn_nvrtc_cubin_with_fmad ? "true" : "false";
    compile_options.push_back("--fmad=" + enable_fmad);
  } else {
    compile_options.push_back("-arch=compute_" + cc);
  }
  compile_options.push_back("-std=c++14");
  compile_options.push_back("-default-device");

  if (include_headers) {  // prepare include headers
    auto cuda_headers = FindCUDAIncludePaths();
    auto cinn_headers = FindCINNRuntimeIncludePaths();
    std::vector<std::string> include_paths;
    for (auto& header : cuda_headers) {
      VLOG(5) << "add include-path: " << header;
      include_paths.push_back("--include-path=" + header);
    }
    for (auto& header : cinn_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    compile_options.insert(
        std::end(compile_options), include_paths.begin(), include_paths.end());
  }

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }

  std::string fake_code =
      R"(
    extern \"C\" {

__global__
void __launch_bounds__(1) fn_generate_shape_cast_yield_store_broadcast_to_divide_generate_shape_cast_broadcast_to_divide_elementwise_mul_subtract_fill_constant_elementwise_add_rsqrt_generate_shape_broadcast_to_subtract_generate_shape_broadcast_to_elementwise_mul_scale_reshape_scale_broadcast_to_elementwise_add_scale_reshape_scale_broadcast_to_elementwise_add_reshape_reshape_assign_out__assign_out__reshape_generate_shape_broadcast_to_elementwise_mul_reshape_generate_shape_broadcast_to_elementwise_add_yield_store_reshape_yield_store_fill_constant_scale_exp_generate_shape_broadcast_to_elementwise_add_generate_shape_broadcast_to_divide_yield_store_elementwise_mul_yield_store___COND__FPA_trueAND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_LE1023ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_4, const float* __restrict__ var_9, float* __restrict__ var_23, float* __restrict__ var_29, const float* __restrict__ var_38, const float* __restrict__ var_43, float* __restrict__ var_2, float* __restrict__ var_48, float* __restrict__ var_50, float* __restrict__ var_60, float* __restrict__ var_62, int64_t S0, int64_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((S0 * S1) * 512ll)));
  float* var_36 = var_23;
  float* var_37 = var_29;
  if ((((int)blockIdx.x % ((S0 * S1) * 512ll)) == 0ll)) {
    var_2[((int)blockIdx.x / ((S0 * S1) * 512ll))] = ((float)((S0 * S1)));
  };
  if ((((int)blockIdx.x % (S0 * S1)) == 0ll)) {
    var_36[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] = ((0.899999976f * var_23[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]) + (0.100000024f * (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))));
  };
  if ((((int)blockIdx.x % (S0 * S1)) == 0ll)) {
    var_37[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] = ((0.899999976f * var_29[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]) + (0.100000024f * ((var_9[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))))));
  };
  if ((((int)blockIdx.x % (S0 * S1)) == 0ll)) {
    var_50[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] = cinn_nvgpu_rsqrt_fp32((((var_9[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))))) + 9.99999975e-06f));
  };
  var_48[(((((((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((int)blockIdx.x / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((int)blockIdx.x % (S0 * S1)))] = ((((var[(((((((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((int)blockIdx.x / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((int)blockIdx.x % (S0 * S1)))] - (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]) + var_43[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]);
  var_60[(((((((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((int)blockIdx.x / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((int)blockIdx.x % (S0 * S1)))] = (1.00000000f / (1.00000000f + cinn_nvgpu_exp_fp32((-1.00000000f * ((((var[(((((((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((int)blockIdx.x / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((int)blockIdx.x % (S0 * S1)))] - (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]) + var_43[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))])))));
  var_62[(((((((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((int)blockIdx.x / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((int)blockIdx.x % (S0 * S1)))] = (((((var[(((((((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((int)blockIdx.x / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((int)blockIdx.x % (S0 * S1)))] - (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]) + var_43[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]) * (1.00000000f / (1.00000000f + cinn_nvgpu_exp_fp32((-1.00000000f * ((((var[(((((((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((int)blockIdx.x / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((int)blockIdx.x % (S0 * S1)))] - (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]) + var_43[(((int)blockIdx.x % ((S0 * S1) * 512ll)) / (S0 * S1))]))))));
}__global__
void __launch_bounds__(1024) fn_generate_shape_cast_yield_store_broadcast_to_divide_generate_shape_cast_broadcast_to_divide_elementwise_mul_subtract_fill_constant_elementwise_add_rsqrt_generate_shape_broadcast_to_subtract_generate_shape_broadcast_to_elementwise_mul_scale_reshape_scale_broadcast_to_elementwise_add_scale_reshape_scale_broadcast_to_elementwise_add_reshape_reshape_assign_out__assign_out__reshape_generate_shape_broadcast_to_elementwise_mul_reshape_generate_shape_broadcast_to_elementwise_add_yield_store_reshape_yield_store_fill_constant_scale_exp_generate_shape_broadcast_to_elementwise_add_generate_shape_broadcast_to_divide_yield_store_elementwise_mul_yield_store___COND__FPA_trueAND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_GE1024ll_BPA_AND_FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_LE1048575ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_4, const float* __restrict__ var_9, float* __restrict__ var_23, float* __restrict__ var_29, const float* __restrict__ var_38, const float* __restrict__ var_43, float* __restrict__ var_2, float* __restrict__ var_48, float* __restrict__ var_50, float* __restrict__ var_60, float* __restrict__ var_62, int64_t S0, int64_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((((S0 * S1) * 512ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  float* var_36 = var_23;
  float* var_37 = var_29;
  for (int32_t i_append_var_67_append_var_68_append_var_69_fused_0 = 0ll; i_append_var_67_append_var_68_append_var_69_fused_0 < 4ll; i_append_var_67_append_var_68_append_var_69_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_append_var_67_append_var_68_append_var_69_fused_0) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + i_append_var_67_append_var_68_append_var_69_fused_0) * 1024ll) + (int)threadIdx.x) % ((S0 * S1) * 512ll)) == 0ll)) {
        var_2[((((i_append_var_67_append_var_68_append_var_69_fused_0 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll))] = ((float)((S0 * S1)));
      };
    };
  };
  for (int32_t append_var_64_i_append_var_65_append_var_66_fused_0 = 0ll; append_var_64_i_append_var_65_append_var_66_fused_0 < 4ll; append_var_64_i_append_var_65_append_var_66_fused_0 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_0) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_0) * 1024ll) + (int)threadIdx.x) % (S0 * S1)) == 0ll)) {
        var_36[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_0 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] = ((0.899999976f * var_23[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_0 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))]) + (0.100000024f * (var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_0 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))));
      };
    };
  };
  for (int32_t append_var_64_i_append_var_65_append_var_66_fused_3 = 0ll; append_var_64_i_append_var_65_append_var_66_fused_3 < 4ll; append_var_64_i_append_var_65_append_var_66_fused_3 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_3) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_3) * 1024ll) + (int)threadIdx.x) % (S0 * S1)) == 0ll)) {
        var_37[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_3 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] = ((0.899999976f * var_29[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_3 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))]) + (0.100000024f * ((var_9[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_3 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_3 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_3 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))))));
      };
    };
  };
  for (int32_t append_var_64_i_append_var_65_append_var_66_fused_6 = 0ll; append_var_64_i_append_var_65_append_var_66_fused_6 < 4ll; append_var_64_i_append_var_65_append_var_66_fused_6 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_6) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_6) * 1024ll) + (int)threadIdx.x) % (S0 * S1)) == 0ll)) {
        var_50[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_6 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] = cinn_nvgpu_rsqrt_fp32((((var_9[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_6 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_6 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_6 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))))) + 9.99999975e-06f));
      };
    };
  };
  for (int32_t i_j_k_a_fused_513 = 0ll; i_j_k_a_fused_513 < 4ll; i_j_k_a_fused_513 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_513) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      float var_local_148 = var[(((((((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))];
      float var_4_local_93 = var_4[(((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_9_local_7 = var_9[(((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_38_local_3 = var_38[(((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_43_local_3 = var_43[(((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      var_62[(((((((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))] = (((((var_local_148 - (var_4_local_93 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_7 / ((float)((S0 * S1)))) - ((var_4_local_93 / ((float)((S0 * S1)))) * (var_4_local_93 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_3) + var_43_local_3) * (1.00000000f / (1.00000000f + cinn_nvgpu_exp_fp32((-1.00000000f * ((((var_local_148 - (var_4_local_93 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_7 / ((float)((S0 * S1)))) - ((var_4_local_93 / ((float)((S0 * S1)))) * (var_4_local_93 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_3) + var_43_local_3))))));
      var_48[(((((((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_513 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))] = ((((var_local_148 - (var_4_local_93 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_7 / ((float)((S0 * S1)))) - ((var_4_local_93 / ((float)((S0 * S1)))) * (var_4_local_93 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_3) + var_43_local_3);
    };
  };
  for (int32_t i_j_k_a_fused_516 = 0ll; i_j_k_a_fused_516 < 4ll; i_j_k_a_fused_516 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_516) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      float var_local_149 = var[(((((((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))];
      float var_4_local_94 = var_4[(((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_9_local_8 = var_9[(((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_38_local_4 = var_38[(((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_43_local_4 = var_43[(((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      var_60[(((((((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_516 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))] = (1.00000000f / (1.00000000f + cinn_nvgpu_exp_fp32((-1.00000000f * ((((var_local_149 - (var_4_local_94 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_8 / ((float)((S0 * S1)))) - ((var_4_local_94 / ((float)((S0 * S1)))) * (var_4_local_94 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_4) + var_43_local_4)))));
    };
  };
}__global__
void __launch_bounds__(1024) fn_generate_shape_cast_yield_store_broadcast_to_divide_generate_shape_cast_broadcast_to_divide_elementwise_mul_subtract_fill_constant_elementwise_add_rsqrt_generate_shape_broadcast_to_subtract_generate_shape_broadcast_to_elementwise_mul_scale_reshape_scale_broadcast_to_elementwise_add_scale_reshape_scale_broadcast_to_elementwise_add_reshape_reshape_assign_out__assign_out__reshape_generate_shape_broadcast_to_elementwise_mul_reshape_generate_shape_broadcast_to_elementwise_add_yield_store_reshape_yield_store_fill_constant_scale_exp_generate_shape_broadcast_to_elementwise_add_generate_shape_broadcast_to_divide_yield_store_elementwise_mul_yield_store___COND__FPA_trueAND_FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_GE1048576ll_BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_4, const float* __restrict__ var_9, float* __restrict__ var_23, float* __restrict__ var_29, const float* __restrict__ var_38, const float* __restrict__ var_43, float* __restrict__ var_2, float* __restrict__ var_48, float* __restrict__ var_50, float* __restrict__ var_60, float* __restrict__ var_62, int64_t S0, int64_t S1)
{
  __builtin_assume(((int)blockIdx.x < ((((S0 * S1) * 512ll) / 4096ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 1024ll));
  float* var_36 = var_23;
  float* var_37 = var_29;
  for (int32_t i_append_var_67_append_var_68_append_var_69_fused_6 = 0ll; i_append_var_67_append_var_68_append_var_69_fused_6 < 4ll; i_append_var_67_append_var_68_append_var_69_fused_6 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_append_var_67_append_var_68_append_var_69_fused_6) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + i_append_var_67_append_var_68_append_var_69_fused_6) * 1024ll) + (int)threadIdx.x) % ((S0 * S1) * 512ll)) == 0ll)) {
        var_2[((((i_append_var_67_append_var_68_append_var_69_fused_6 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll))] = ((float)((S0 * S1)));
      };
    };
  };
  for (int32_t append_var_64_i_append_var_65_append_var_66_fused_9 = 0ll; append_var_64_i_append_var_65_append_var_66_fused_9 < 4ll; append_var_64_i_append_var_65_append_var_66_fused_9 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_9) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_9) * 1024ll) + (int)threadIdx.x) % (S0 * S1)) == 0ll)) {
        var_36[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_9 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] = ((0.899999976f * var_23[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_9 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))]) + (0.100000024f * (var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_9 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))));
      };
    };
  };
  for (int32_t append_var_64_i_append_var_65_append_var_66_fused_12 = 0ll; append_var_64_i_append_var_65_append_var_66_fused_12 < 4ll; append_var_64_i_append_var_65_append_var_66_fused_12 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_12) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_12) * 1024ll) + (int)threadIdx.x) % (S0 * S1)) == 0ll)) {
        var_37[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_12 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] = ((0.899999976f * var_29[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_12 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))]) + (0.100000024f * ((var_9[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_12 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_12 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_12 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1))))))));
      };
    };
  };
  for (int32_t append_var_64_i_append_var_65_append_var_66_fused_15 = 0ll; append_var_64_i_append_var_65_append_var_66_fused_15 < 4ll; append_var_64_i_append_var_65_append_var_66_fused_15 += 1) {
    if (((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_15) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      if ((((((((int)blockIdx.x * 4ll) + append_var_64_i_append_var_65_append_var_66_fused_15) * 1024ll) + (int)threadIdx.x) % (S0 * S1)) == 0ll)) {
        var_50[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_15 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] = cinn_nvgpu_rsqrt_fp32((((var_9[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_15 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) - ((var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_15 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))) * (var_4[((((((int)blockIdx.x * 4096ll) + (int)threadIdx.x) + (append_var_64_i_append_var_65_append_var_66_fused_15 * 1024ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))] / ((float)((S0 * S1)))))) + 9.99999975e-06f));
      };
    };
  };
  for (int32_t i_j_k_a_fused_528 = 0ll; i_j_k_a_fused_528 < 4ll; i_j_k_a_fused_528 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_528) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      float var_local_150 = var[(((((((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))];
      float var_4_local_95 = var_4[(((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_9_local_9 = var_9[(((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_38_local_5 = var_38[(((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_43_local_5 = var_43[(((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      var_62[(((((((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))] = (((((var_local_150 - (var_4_local_95 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_9 / ((float)((S0 * S1)))) - ((var_4_local_95 / ((float)((S0 * S1)))) * (var_4_local_95 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_5) + var_43_local_5) * (1.00000000f / (1.00000000f + cinn_nvgpu_exp_fp32((-1.00000000f * ((((var_local_150 - (var_4_local_95 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_9 / ((float)((S0 * S1)))) - ((var_4_local_95 / ((float)((S0 * S1)))) * (var_4_local_95 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_5) + var_43_local_5))))));
      var_48[(((((((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_528 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))] = ((((var_local_150 - (var_4_local_95 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_9 / ((float)((S0 * S1)))) - ((var_4_local_95 / ((float)((S0 * S1)))) * (var_4_local_95 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_5) + var_43_local_5);
    };
  };
  for (int32_t i_j_k_a_fused_531 = 0ll; i_j_k_a_fused_531 < 4ll; i_j_k_a_fused_531 += 1) {
    if (((((((int)blockIdx.x * 4ll) + i_j_k_a_fused_531) * 1024ll) + (int)threadIdx.x) < ((S0 * S1) * 512ll))) {
      float var_local_151 = var[(((((((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))];
      float var_4_local_96 = var_4[(((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_9_local_10 = var_9[(((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_38_local_6 = var_38[(((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      float var_43_local_6 = var_43[(((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1))];
      var_60[(((((((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % ((S0 * S1) * 512ll)) / (S0 * S1)) + (((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) / ((S0 * S1) * 512ll)) * 512ll)) * S0) * S1) + ((((i_j_k_a_fused_531 * 1024ll) + (int)threadIdx.x) + ((int)blockIdx.x * 4096ll)) % (S0 * S1)))] = (1.00000000f / (1.00000000f + cinn_nvgpu_exp_fp32((-1.00000000f * ((((var_local_151 - (var_4_local_96 / ((float)((S0 * S1))))) * cinn_nvgpu_rsqrt_fp32((((var_9_local_10 / ((float)((S0 * S1)))) - ((var_4_local_96 / ((float)((S0 * S1)))) * (var_4_local_96 / ((float)((S0 * S1)))))) + 9.99999975e-06f))) * var_38_local_6) + var_43_local_6)))));
    };
  };
}

})";

  // std::string fake_code =
  //   R"(
  //     extern \"C\" {

  // __global__
  // void __launch_bounds__(1)
  // fn_generate_shape_cast_yield_store_broadcast_to_divide_generate_shape_cast_broadcast_to_divide_elementwise_mul_subtract_fill_constant_elementwise_add_rsqrt_generate_shape_broadcast_to_subtract_generate_shape_broadcast_to_elementwise_mul_scale_reshape_scale_broadcast_to_elementwise_add_scale_reshape_scale_broadcast_to_elementwise_add_reshape_reshape_assign_out__assign_out__reshape_generate_shape_broadcast_to_elementwise_mul_reshape_generate_shape_broadcast_to_elementwise_add_yield_store_reshape_yield_store_fill_constant_scale_exp_generate_shape_broadcast_to_elementwise_add_generate_shape_broadcast_to_divide_yield_store_elementwise_mul_yield_store___COND__FPA_trueAND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_LE1023ll_BPA__BPA__BPA___kernel(const
  // float* __restrict__ var, const float* __restrict__ var_4, const float*
  // __restrict__ var_9, float* __restrict__ var_23, float* __restrict__ var_29,
  // const float* __restrict__ var_38, const float* __restrict__ var_43, float*
  // __restrict__ var_2, float* __restrict__ var_48, float* __restrict__ var_50,
  // float* __restrict__ var_60, float* __restrict__ var_62, int64_t S0, int64_t
  // S1)
  // {

  // }

  // })";

  VLOG(0) << "================= header time 1================";

  VLOG(3) << "compile options: " << utils::Join(compile_options, " ");

  nvrtcResult compile_res;

  for (int i = 0; i < 2000; ++i) {
    NVRTC_CALL(nvrtcCreateProgram(&prog,
                                  fake_code.c_str(),
                                  nullptr,
                                  header_gen.size(),
                                  header_gen.headers().data(),
                                  header_gen.include_names().data()));

    compile_res =
        nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());
  }
  VLOG(0) << "================= header time 2================";

  nvrtcProgram prog1;

  for (int i = 0; i < 2000; ++i) {
    NVRTC_CALL(nvrtcCreateProgram(
        &prog1, fake_code.c_str(), nullptr, 0, nullptr, nullptr));

    compile_res = nvrtcCompileProgram(
        prog1, param_cstrings.size(), param_cstrings.data());
  }
  VLOG(0) << "================= header time 3================";

  fake_code =
      R"(
    extern \"C\" {

__global__
void __launch_bounds__(1) fn_generate_shape_cast_yield_store_broadcast_to_divide_generate_shape_cast_broadcast_to_divide_elementwise_mul_subtract_fill_constant_elementwise_add_rsqrt_generate_shape_broadcast_to_subtract_generate_shape_broadcast_to_elementwise_mul_scale_reshape_scale_broadcast_to_elementwise_add_scale_reshape_scale_broadcast_to_elementwise_add_reshape_reshape_assign_out__assign_out__reshape_generate_shape_broadcast_to_elementwise_mul_reshape_generate_shape_broadcast_to_elementwise_add_yield_store_reshape_yield_store_fill_constant_scale_exp_generate_shape_broadcast_to_elementwise_add_generate_shape_broadcast_to_divide_yield_store_elementwise_mul_yield_store___COND__FPA_trueAND_FPA__FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_GE1ll_BPA_AND_FPA__FPA__FPA_S0MULS1_BPA_MUL512ll_BPA_LE1023ll_BPA__BPA__BPA___kernel(const float* __restrict__ var, const float* __restrict__ var_4, const float* __restrict__ var_9, float* __restrict__ var_23, float* __restrict__ var_29, const float* __restrict__ var_38, const float* __restrict__ var_43, float* __restrict__ var_2, float* __restrict__ var_48, float* __restrict__ var_50, float* __restrict__ var_60, float* __restrict__ var_62, int64_t S0, int64_t S1)
{

}

})";

  VLOG(0) << "================= header time 4================";

  for (int i = 0; i < 2000; ++i) {
    NVRTC_CALL(nvrtcCreateProgram(&prog,
                                  fake_code.c_str(),
                                  nullptr,
                                  header_gen.size(),
                                  header_gen.headers().data(),
                                  header_gen.include_names().data()));

    compile_res =
        nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());
  }
  VLOG(0) << "================= header time 5================";

  for (int i = 0; i < 2000; ++i) {
    NVRTC_CALL(nvrtcCreateProgram(
        &prog1, fake_code.c_str(), nullptr, 0, nullptr, nullptr));

    compile_res = nvrtcCompileProgram(
        prog1, param_cstrings.size(), param_cstrings.data());
  }
  VLOG(0) << "================= header time 6================";

  std::stringstream ss;
  ss << "====================================.";
  PADDLE_THROW(::common::errors::Fatal(ss.str()));

  {  // get log
    size_t log_size;
    NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));

    PADDLE_ENFORCE_EQ(
        compile_res,
        NVRTC_SUCCESS,
        ::common::errors::Fatal("NVRTC compilation failed: %s", log));
  }

  size_t size;
  std::string data;
  if (compile_to_cubin_) {
    NVRTC_CALL(nvrtcGetCUBINSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetCUBIN(prog, &data[0]));
  } else {
    NVRTC_CALL(nvrtcGetPTXSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetPTX(prog, &data[0]));
  }

  NVRTC_CALL(nvrtcDestroyProgram(&prog));
  return data;
}

std::string Compiler::CompileWithNvcc(const std::string& cuda_c) {
  // read dir source
  std::string dir = "./source";
  if (access(dir.c_str(), 0) == -1) {
    PADDLE_ENFORCE_NE(
        mkdir(dir.c_str(), 7),
        -1,
        ::common::errors::PermissionDenied(
            "Failed to create directory %s. Please check the permissions.",
            dir.c_str()));
  }

  // get unique prefix name
  prefix_name_ = dir + "/" + cinn::common::UniqName("rtc_tmp");

  auto cuda_c_file = prefix_name_ + ".cu";
  std::ofstream ofs(cuda_c_file, std::ios::out);
  PADDLE_ENFORCE_EQ(ofs.is_open(),
                    true,
                    ::common::errors::Unavailable(
                        "Failed to open file %s. Please check if the file path "
                        "is correct and the file is accessible.",
                        cuda_c_file.c_str()));
  ofs << cuda_c;
  ofs.close();

  CompileToPtx();
  CompileToCubin();

  return prefix_name_ + ".cubin";
}

// std::string Compiler::GetPtx() { return ReadFile(prefix_name_ + ".ptx",
// std::ios::in); }

void Compiler::CompileToPtx() {
  auto include_dir = cinn::common::Context::Global().runtime_include_dir();
  std::string include_dir_str = "";
  for (auto dir : include_dir) {
    if (include_dir_str.empty()) {
      include_dir_str = dir;
    } else {
      include_dir_str += ":" + dir;
    }
  }

  std::string options = std::string("export PATH=") + FLAGS_cinn_nvcc_cmd_path +
                        std::string(":$PATH && nvcc -std=c++14 --ptx -O3 -I ") +
                        include_dir_str;
  options += " -arch=" + GetDeviceArch();
  options += " -o " + prefix_name_ + ".ptx";
  options += " " + prefix_name_ + ".cu";

  VLOG(2) << "Nvcc Compile Options : " << options;
  PADDLE_ENFORCE_EQ(
      system(options.c_str()),
      0,
      ::common::errors::InvalidArgument("Failed to execute command: %s. Please "
                                        "check the command and try again.",
                                        options.c_str()));
}

void Compiler::CompileToCubin() {
  std::string options = std::string("export PATH=") + FLAGS_cinn_nvcc_cmd_path +
                        std::string(":$PATH && nvcc --cubin -O3");
  options += " -arch=" + GetDeviceArch();
  options += " -o " + prefix_name_ + ".cubin";
  options += " " + prefix_name_ + ".ptx";

  VLOG(2) << "Nvcc Compile Options : " << options;
  PADDLE_ENFORCE_EQ(
      system(options.c_str()),
      0,
      ::common::errors::InvalidArgument("Failed to execute command: %s. Please "
                                        "check the command and try again.",
                                        options.c_str()));
}

std::string Compiler::GetDeviceArch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) ==
          cudaSuccess &&
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) ==
          cudaSuccess) {
    return "sm_" + std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
    return "sm_30";
  }
}

std::string Compiler::ReadFile(const std::string& file_name,
                               std::ios_base::openmode mode) {
  // open cubin file
  std::ifstream ifs(file_name, mode);
  PADDLE_ENFORCE_EQ(ifs.is_open(),
                    true,
                    ::common::errors::Unavailable(
                        "Failed to open file %s. Please check if the file path "
                        "is correct and the file is accessible.",
                        file_name.c_str()));
  ifs.seekg(std::ios::end);
  auto len = ifs.tellg();
  ifs.seekg(0);

  // read cubin file
  std::string file_data(len, ' ');
  ifs.read(&file_data[0], len);
  ifs.close();
  return std::move(file_data);
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
