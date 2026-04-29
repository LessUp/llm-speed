---
layout: default
title: LLM-Speed
description: CUDA kernel library for LLM inference with FlashAttention forward, Tensor Core GEMM, and PyTorch bindings. Optimized for Ampere and newer architectures.
lang: en
---

<!-- Hero Section -->
<section class="hero">
  <div class="hero-content">
    <div class="hero-badge">
      <span class="badge-dot"></span>
      <span>v{{ site.current_version }} — CUDA kernels for LLM inference</span>
    </div>
    <h1 class="hero-title">LLM-Speed</h1>
    <p class="hero-description">
      A focused CUDA kernel library implementing FlashAttention forward, Tensor Core GEMM acceleration,
      and seamless PyTorch integration. Designed for efficient LLM inference on modern GPUs.
    </p>
    <div class="hero-actions">
      <a href="{{ site.baseurl }}/docs/setup/quickstart-en/" class="btn btn-primary">
        🚀 Get Started
      </a>
      <a href="{{ site.baseurl }}/docs/architecture/architecture-en/" class="btn btn-secondary">
        📖 Understand Architecture
      </a>
      <a href="{{ site.baseurl }}/docs/tutorials/performance-en/" class="btn btn-secondary">
        📊 Benchmark Locally
      </a>
    </div>
    <div class="badges">
      <span class="badge"><img src="https://github.com/LessUp/llm-speed/actions/workflows/ci.yml/badge.svg" alt="CI"></span>
      <span class="badge"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></span>
      <span class="badge"><img src="https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white" alt="CUDA"></span>
      <span class="badge"><img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white" alt="C++"></span>
      <span class="badge"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white" alt="Python"></span>
    </div>
  </div>
</section>

<!-- Features Section -->
<section class="features">
  <div class="section-header">
    <h2 class="section-title">Key Features</h2>
    <p class="section-description">
      Optimized CUDA kernels for modern LLM inference with memory-efficient algorithms and hardware acceleration
    </p>
  </div>
  <div class="feature-grid">
    <div class="feature-card">
      <div class="feature-icon">⚡</div>
      <h3>FlashAttention</h3>
      <p>O(N) memory complexity with online softmax algorithm. Supports causal masking for autoregressive models.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🔢</div>
      <h3>Tensor Core GEMM</h3>
      <p>Hardware-accelerated matrix multiplication using WMMA API. FP16 input with FP32 accumulation.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🐍</div>
      <h3>PyTorch Integration</h3>
      <p>Seamless integration with PyTorch via pybind11. Native CUDA tensor support.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🔄</div>
      <h3>Double Buffering</h3>
      <p>Compute/memory overlap with pipelined execution. Async copy for Ampere+ architectures.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🏦</div>
      <h3>Bank Conflict Free</h3>
      <p>Carefully designed shared memory layouts with padding to eliminate bank conflicts.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">📊</div>
      <h3>Property Testing</h3>
      <p>Comprehensive tests with Hypothesis for correctness verification across edge cases.</p>
    </div>
  </div>
</section>

<!-- Memory Efficiency Design -->
<section class="performance-section">
  <div class="section-header">
    <h2 class="section-title">Memory-Efficient Design</h2>
    <p class="section-description">
      FlashAttention implements online softmax with O(N) memory complexity instead of O(N²) for standard attention.
    </p>
  </div>
  <div class="gpu-table">
    <table>
      <thead>
        <tr>
          <th>Sequence Length</th>
          <th>Standard Attention</th>
          <th>FlashAttention</th>
          <th>Memory Savings</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1024</td>
          <td>4 MB (full attention matrix)</td>
          <td>0.25 MB (streaming)</td>
          <td>16×</td>
        </tr>
        <tr>
          <td>4096</td>
          <td>64 MB (full attention matrix)</td>
          <td>1 MB (streaming)</td>
          <td>64×</td>
        </tr>
        <tr>
          <td>8192</td>
          <td>256 MB (full attention matrix)</td>
          <td>2 MB (streaming)</td>
          <td>128×</td>
        </tr>
      </tbody>
    </table>
  </div>
  <p class="section-description" style="margin-top: 1.5rem; color: var(--text-secondary);">
    Assumes 8 attention heads, FP32 accumulation, batch size 1. Exact savings depend on hardware and kernel implementation.
  </p>
</section>

<!-- Code Preview -->
<section class="code-preview">
  <div class="section-header">
    <h2 class="section-title">Quick Example</h2>
    <p class="section-description">Get started with just a few lines of code</p>
  </div>
  <div class="code-preview-grid">
    <div class="code-preview-card">
      <div class="code-preview-header">
        <span class="code-dot red"></span>
        <span class="code-dot yellow"></span>
        <span class="code-dot green"></span>
        <span>flash_attention.py</span>
      </div>
      <pre><code class="language-python">import torch
from cuda_llm_ops import flash_attention

# Create inputs
batch, heads = 2, 8
seq_len, head_dim = 2048, 64

q = torch.randn(batch, heads, seq_len, head_dim,
                device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# O(N) memory attention!
output = flash_attention(q, k, v, is_causal=True)</code></pre>
    </div>
    <div class="code-preview-card">
      <div class="code-preview-header">
        <span class="code-dot red"></span>
        <span class="code-dot yellow"></span>
        <span class="code-dot green"></span>
        <span>tensor_core_gemm.py</span>
      </div>
      <pre><code class="language-python">import torch
from cuda_llm_ops import tensor_core_gemm

# Matrix multiplication
a = torch.randn(1024, 512, device='cuda',
                dtype=torch.float16)
b = torch.randn(512, 1024, device='cuda',
                dtype=torch.float16)

# Hardware accelerated GEMM
# FP16 input → FP32 output
c = tensor_core_gemm(a, b)
print(c.dtype)  # torch.float32</code></pre>
    </div>
  </div>
</section>

<!-- Architecture Support -->
<section class="features" style="background: var(--bg-surface); padding: 4rem 0;">
  <div class="section-header">
    <h2 class="section-title">GPU Architecture Support</h2>
    <p class="section-description">Optimized for Ampere (A100, RTX 30) and newer. Forward compatibility with Hopper and future architectures.</p>
  </div>
  <div class="gpu-table">
    <table>
      <thead>
        <tr>
          <th>Architecture</th>
          <th>Tensor Core</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Ampere</strong> (A100, RTX 30/40)</td>
          <td>WMMA with FP16, BF16, TF32</td>
          <td>✅ Primary target</td>
        </tr>
        <tr>
          <td><strong>Hopper</strong> (H100)</td>
          <td>WMMA with FP16, BF16, FP8</td>
          <td>✅ Supported</td>
        </tr>
        <tr>
          <td><strong>Volta</strong> (V100)</td>
          <td>WMMA with FP16</td>
          <td>⚠️ Limited</td>
        </tr>
        <tr>
          <td><strong>Turing</strong> (T4, RTX 20)</td>
          <td>WMMA with FP16, INT8</td>
          <td>⚠️ Limited</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<!-- Documentation Cards -->
<section class="features" style="background: var(--bg-body);">
  <div class="section-header">
    <h2 class="section-title">Documentation</h2>
    <p class="section-description">Comprehensive guides in English and Chinese</p>
  </div>
  <div class="feature-grid" style="max-width: 1000px;">
    <a href="{{ site.baseurl }}/docs/setup/quickstart-en/" class="feature-card" style="text-decoration: none; color: inherit;">
      <div class="feature-icon">🚀</div>
      <h3>Quick Start</h3>
      <p>Get up and running in 5 minutes with installation and basic usage examples.</p>
    </a>
    <a href="{{ site.baseurl }}/docs/api/api-en/" class="feature-card" style="text-decoration: none; color: inherit;">
      <div class="feature-icon">📚</div>
      <h3>API Reference</h3>
      <p>Complete API documentation with parameters, examples, and error handling.</p>
    </a>
    <a href="{{ site.baseurl }}/docs/architecture/architecture-en/" class="feature-card" style="text-decoration: none; color: inherit;">
      <div class="feature-icon">🏗️</div>
      <h3>Architecture</h3>
      <p>Technical deep dive into CUDA kernels, optimization strategies, and implementation details.</p>
    </a>
    <a href="{{ site.baseurl }}/docs/tutorials/performance-en/" class="feature-card" style="text-decoration: none; color: inherit;">
      <div class="feature-icon">⚡</div>
      <h3>Performance Guide</h3>
      <p>Optimization tips, benchmarking tools, and best practices for maximum performance.</p>
    </a>
  </div>
</section>

<!-- Next Steps -->
<section class="cta">
  <h2 class="cta-title">Start using LLM-Speed</h2>
  <p class="cta-description">
    Three clear paths to get value from this project:
  </p>
  <div class="feature-grid" style="max-width: 900px; margin: 2rem auto; gap: 1.5rem;">
    <a href="{{ site.baseurl }}/docs/setup/quickstart-en/" class="feature-card" style="text-decoration: none; color: inherit; border: 1px solid var(--border-color); border-radius: 8px; padding: 1.5rem;">
      <div class="feature-icon" style="font-size: 2rem;">🚀</div>
      <h3>Get Started</h3>
      <p>Install and run your first FlashAttention or Tensor Core GEMM example in 5 minutes.</p>
    </a>
    <a href="{{ site.baseurl }}/docs/architecture/architecture-en/" class="feature-card" style="text-decoration: none; color: inherit; border: 1px solid var(--border-color); border-radius: 8px; padding: 1.5rem;">
      <div class="feature-icon" style="font-size: 2rem;">🏗️</div>
      <h3>Understand Architecture</h3>
      <p>Explore kernel design, memory layout optimization, and Tensor Core utilization patterns.</p>
    </a>
    <a href="{{ site.baseurl }}/docs/tutorials/performance-en/" class="feature-card" style="text-decoration: none; color: inherit; border: 1px solid var(--border-color); border-radius: 8px; padding: 1.5rem;">
      <div class="feature-icon" style="font-size: 2rem;">📊</div>
      <h3>Benchmark Locally</h3>
      <p>Run performance benchmarks on your GPU. See memory usage and speedups with provided tools.</p>
    </a>
  </div>
  <div class="hero-actions" style="margin-top: 2rem;">
    <a href="{{ site.github.repository_url }}" class="btn btn-secondary" target="_blank" rel="noopener">
      💻 Browse Source
    </a>
    <a href="{{ site.baseurl }}/docs/api/api-en/" class="btn btn-secondary">
      📚 API Reference
    </a>
  </div>
</section>
