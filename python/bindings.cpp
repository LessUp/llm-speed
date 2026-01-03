#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.cuh"

namespace py = pybind11;

// Forward declarations
void naive_attention_fp32(const float*, const float*, const float*, float*,
                          int, int, int, int, float, cudaStream_t);
void naive_attention_fp16(const half*, const half*, const half*, half*,
                          int, int, int, int, float, cudaStream_t);
void tiled_attention_fp32(const float*, const float*, const float*, float*,
                          int, int, int, int, float, cudaStream_t);
void tiled_attention_fp16(const half*, const half*, const half*, half*,
                          int, int, int, int, float, cudaStream_t);
void flash_attention_fp32(const float*, const float*, const float*, float*,
                          int, int, int, int, float, bool, cudaStream_t);
void flash_attention_fp16(const half*, const half*, const half*, half*,
                          int, int, int, int, float, bool, cudaStream_t);
void tensor_core_gemm_fp16(const half*, const half*, float*,
                           int, int, int, float, float, cudaStream_t);
void hgemm_fp32(const float*, const float*, float*,
                int, int, int, float, float,
                MatrixLayout, MatrixLayout, cudaStream_t);
void hgemm_fp16(const half*, const half*, half*,
                int, int, int, float, float,
                MatrixLayout, MatrixLayout, cudaStream_t);

// Input validation
void validate_attention_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    TORCH_CHECK(q.dim() == 4, "Q must be 4D tensor [batch, heads, seq_len, head_dim]");
    TORCH_CHECK(k.dim() == 4, "K must be 4D tensor [batch, heads, seq_len, head_dim]");
    TORCH_CHECK(v.dim() == 4, "V must be 4D tensor [batch, heads, seq_len, head_dim]");
    
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have same shape");
    TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have same shape");
    
    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V must be on CUDA device");
    
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
    
    auto dtype = q.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16,
                "Only float32 and float16 are supported");
    TORCH_CHECK(k.scalar_type() == dtype, "K must have same dtype as Q");
    TORCH_CHECK(v.scalar_type() == dtype, "V must have same dtype as Q");
}

void validate_gemm_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    TORCH_CHECK(a.dim() == 2, "A must be 2D tensor [M, K]");
    TORCH_CHECK(b.dim() == 2, "B must be 2D tensor [K, N]");
    
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match: A[M,K] @ B[K,N]");
    
    TORCH_CHECK(a.is_cuda(), "A must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "B must be on CUDA device");
    
    TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "B must be contiguous");
}

// Naive attention wrapper
torch::Tensor naive_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    float scale = 0.0f
) {
    validate_attention_inputs(q, k, v);
    
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    auto output = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (q.scalar_type() == torch::kFloat32) {
        naive_attention_fp32(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            scale, stream
        );
    } else {
        naive_attention_fp16(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            batch_size, num_heads, seq_len, head_dim,
            scale, stream
        );
    }
    
    return output;
}

// Tiled attention wrapper
torch::Tensor tiled_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    float scale = 0.0f
) {
    validate_attention_inputs(q, k, v);
    
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    auto output = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (q.scalar_type() == torch::kFloat32) {
        tiled_attention_fp32(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            scale, stream
        );
    } else {
        tiled_attention_fp16(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            batch_size, num_heads, seq_len, head_dim,
            scale, stream
        );
    }
    
    return output;
}

// Flash attention wrapper
torch::Tensor flash_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    float scale = 0.0f,
    bool is_causal = false
) {
    validate_attention_inputs(q, k, v);
    
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    auto output = torch::empty_like(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (q.scalar_type() == torch::kFloat32) {
        flash_attention_fp32(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            scale, is_causal, stream
        );
    } else {
        flash_attention_fp16(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            batch_size, num_heads, seq_len, head_dim,
            scale, is_causal, stream
        );
    }
    
    return output;
}

// GEMM wrapper
torch::Tensor gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    float alpha = 1.0f,
    float beta = 0.0f,
    bool trans_a = false,
    bool trans_b = false
) {
    validate_gemm_inputs(a, b);
    
    int M = trans_a ? a.size(1) : a.size(0);
    int K = trans_a ? a.size(0) : a.size(1);
    int N = trans_b ? b.size(0) : b.size(1);
    
    auto output = torch::empty({M, N}, a.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    MatrixLayout layout_a = trans_a ? MatrixLayout::ColMajor : MatrixLayout::RowMajor;
    MatrixLayout layout_b = trans_b ? MatrixLayout::ColMajor : MatrixLayout::RowMajor;
    
    if (a.scalar_type() == torch::kFloat32) {
        hgemm_fp32(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, K, alpha, beta,
            layout_a, layout_b, stream
        );
    } else if (a.scalar_type() == torch::kFloat16) {
        hgemm_fp16(
            reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, N, K, alpha, beta,
            layout_a, layout_b, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for GEMM");
    }
    
    return output;
}

// Tensor Core GEMM wrapper (FP16 input, FP32 output)
torch::Tensor tensor_core_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    float alpha = 1.0f,
    float beta = 0.0f
) {
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "Tensor Core GEMM requires FP16 input");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "Tensor Core GEMM requires FP16 input");
    validate_gemm_inputs(a, b);
    
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(a.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    tensor_core_gemm_fp16(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        output.data_ptr<float>(),
        M, N, K, alpha, beta, stream
    );
    
    return output;
}

PYBIND11_MODULE(cuda_llm_ops, m) {
    m.doc() = "CUDA LLM Kernel Optimization - High-performance attention and GEMM kernels";
    
    // Attention functions
    m.def("naive_attention", &naive_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f,
          "Naive attention implementation (baseline)");
    
    m.def("tiled_attention", &tiled_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = 0.0f,
          "Tiled attention with shared memory optimization");
    
    m.def("flash_attention", &flash_attention,
          py::arg("q"), py::arg("k"), py::arg("v"), 
          py::arg("scale") = 0.0f, py::arg("is_causal") = false,
          "FlashAttention with online softmax");
    
    // GEMM functions
    m.def("gemm", &gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("trans_a") = false, py::arg("trans_b") = false,
          "High-performance GEMM with register tiling");
    
    m.def("tensor_core_gemm", &tensor_core_gemm,
          py::arg("a"), py::arg("b"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          "Tensor Core GEMM (FP16 input, FP32 output)");
}
