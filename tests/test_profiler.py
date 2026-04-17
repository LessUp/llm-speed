"""
Tests for the profiler module.
"""

import pytest
import torch

try:
    import cuda_llm_ops
except ImportError:
    pytest.skip("CUDA kernels not built", allow_module_level=True)

from cuda_llm_ops.profiler import (
    Bottleneck,
    CUDAProfiler,
    KernelMetrics,
    benchmark_attention,
    benchmark_gemm,
    print_benchmark_results,
)


@pytest.fixture
def profiler():
    """Create a CUDAProfiler instance."""
    return CUDAProfiler()


@pytest.fixture
def small_attention_inputs(device):
    """Generate small attention inputs for quick tests."""
    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


@pytest.fixture
def small_gemm_inputs(device):
    """Generate small GEMM inputs for quick tests."""
    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    return a, b


class TestKernelMetrics:
    """Tests for KernelMetrics dataclass."""

    def test_kernel_metrics_creation(self):
        """Test creating KernelMetrics with default values."""
        metrics = KernelMetrics(
            elapsed_ms=1.0,
            tflops=10.0,
            memory_bandwidth_gb=100.0,
        )
        assert metrics.elapsed_ms == 1.0
        assert metrics.tflops == 10.0
        assert metrics.memory_bandwidth_gb == 100.0
        assert metrics.sm_occupancy == 0.0
        assert metrics.l2_hit_rate == 0.0
        assert metrics.bottleneck == Bottleneck.MEMORY_BOUND

    def test_kernel_metrics_with_all_fields(self):
        """Test creating KernelMetrics with all fields."""
        metrics = KernelMetrics(
            elapsed_ms=2.0,
            tflops=20.0,
            memory_bandwidth_gb=200.0,
            sm_occupancy=0.8,
            l2_hit_rate=0.9,
            bottleneck=Bottleneck.COMPUTE_BOUND,
        )
        assert metrics.elapsed_ms == 2.0
        assert metrics.tflops == 20.0
        assert metrics.memory_bandwidth_gb == 200.0
        assert metrics.sm_occupancy == 0.8
        assert metrics.l2_hit_rate == 0.9
        assert metrics.bottleneck == Bottleneck.COMPUTE_BOUND


class TestBottleneck:
    """Tests for Bottleneck enum."""

    def test_bottleneck_values(self):
        """Test Bottleneck enum values."""
        assert Bottleneck.COMPUTE_BOUND.value == "compute_bound"
        assert Bottleneck.MEMORY_BOUND.value == "memory_bound"
        assert Bottleneck.LATENCY_BOUND.value == "latency_bound"


class TestCUDAProfiler:
    """Tests for CUDAProfiler class."""

    @pytest.mark.cuda
    def test_measure_time_simple(self, profiler, device):
        """Test measuring time of a simple function."""

        def simple_func():
            x = torch.randn(100, 100, device=device)
            return x.sum()

        elapsed = profiler.measure_time(simple_func, warmup=2, iterations=5)
        assert elapsed > 0
        assert isinstance(elapsed, float)

    @pytest.mark.cuda
    def test_measure_time_with_args(self, profiler, device):
        """Test measuring time with function arguments."""

        def func_with_args(a, b):
            return torch.matmul(a, b)

        a = torch.randn(32, 32, device=device)
        b = torch.randn(32, 32, device=device)

        elapsed = profiler.measure_time(func_with_args, a, b, warmup=2, iterations=5)
        assert elapsed > 0

    @pytest.mark.cuda
    def test_measure_time_with_kwargs(self, profiler, device):
        """Test measuring time with keyword arguments."""

        def func_with_kwargs(x, scale=1.0):
            return x * scale

        x = torch.randn(100, 100, device=device)

        elapsed = profiler.measure_time(func_with_kwargs, x, scale=2.0, warmup=2, iterations=5)
        assert elapsed > 0

    @pytest.mark.cuda
    def test_profile_attention_fp16(self, profiler):
        """Test profiling attention kernel with FP16."""
        metrics = profiler.profile_attention(
            cuda_llm_ops.flash_attention,
            batch_size=1,
            num_heads=2,
            seq_len=32,
            head_dim=16,
            dtype=torch.float16,
            warmup=2,
            iterations=5,
        )

        assert isinstance(metrics, KernelMetrics)
        assert metrics.elapsed_ms > 0
        assert metrics.tflops > 0
        assert metrics.memory_bandwidth_gb > 0
        assert metrics.bottleneck in [Bottleneck.COMPUTE_BOUND, Bottleneck.MEMORY_BOUND]

    @pytest.mark.cuda
    def test_profile_attention_fp32(self, profiler):
        """Test profiling attention kernel with FP32."""
        metrics = profiler.profile_attention(
            cuda_llm_ops.flash_attention,
            batch_size=1,
            num_heads=2,
            seq_len=32,
            head_dim=16,
            dtype=torch.float32,
            warmup=2,
            iterations=5,
        )

        assert isinstance(metrics, KernelMetrics)
        assert metrics.elapsed_ms > 0

    @pytest.mark.cuda
    def test_profile_gemm_fp16(self, profiler):
        """Test profiling GEMM kernel with FP16."""
        metrics = profiler.profile_gemm(
            cuda_llm_ops.gemm,
            M=64,
            N=64,
            K=64,
            dtype=torch.float16,
            warmup=2,
            iterations=5,
        )

        assert isinstance(metrics, KernelMetrics)
        assert metrics.elapsed_ms > 0
        assert metrics.tflops > 0
        assert metrics.memory_bandwidth_gb > 0

    @pytest.mark.cuda
    def test_profile_gemm_fp32(self, profiler):
        """Test profiling GEMM kernel with FP32."""
        metrics = profiler.profile_gemm(
            cuda_llm_ops.gemm,
            M=64,
            N=64,
            K=64,
            dtype=torch.float32,
            warmup=2,
            iterations=5,
        )

        assert isinstance(metrics, KernelMetrics)
        assert metrics.elapsed_ms > 0

    @pytest.mark.cuda
    def test_compare_with_reference_attention(self, profiler, small_attention_inputs):
        """Test comparing custom kernel with reference implementation."""
        q, k, v = small_attention_inputs

        def custom_attn(q, k, v):
            return cuda_llm_ops.flash_attention(q, k, v)

        def ref_attn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        result = profiler.compare_with_reference(
            custom_attn, ref_attn, q, k, v, warmup=2, iterations=5
        )

        assert "custom_ms" in result
        assert "reference_ms" in result
        assert "speedup" in result
        assert "relative_perf" in result
        assert result["custom_ms"] > 0
        assert result["reference_ms"] > 0
        assert result["speedup"] > 0

    @pytest.mark.cuda
    def test_compare_with_reference_gemm(self, profiler, small_gemm_inputs):
        """Test comparing GEMM kernels."""
        a, b = small_gemm_inputs

        def custom_gemm(a, b):
            return cuda_llm_ops.gemm(a, b)

        def ref_gemm(a, b):
            return torch.matmul(a, b)

        result = profiler.compare_with_reference(
            custom_gemm, ref_gemm, a, b, warmup=2, iterations=5
        )

        assert "custom_ms" in result
        assert "reference_ms" in result
        assert result["custom_ms"] > 0


class TestBenchmarkAttention:
    """Tests for benchmark_attention function."""

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_benchmark_attention_basic(self):
        """Test basic benchmark_attention functionality."""
        results = benchmark_attention(
            seq_lengths=[32, 64],
            batch_size=1,
            num_heads=2,
            head_dim=16,
            dtype=torch.float16,
        )

        assert len(results) == 2
        for result in results:
            assert "seq_len" in result
            assert "custom_ms" in result
            assert "custom_tflops" in result
            assert "reference_ms" in result
            assert "reference_tflops" in result
            assert "speedup" in result
            assert "bottleneck" in result
            assert result["custom_ms"] > 0

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_benchmark_attention_fp32(self):
        """Test benchmark_attention with FP32."""
        results = benchmark_attention(
            seq_lengths=[32],
            batch_size=1,
            num_heads=2,
            head_dim=16,
            dtype=torch.float32,
        )

        assert len(results) == 1
        assert results[0]["seq_len"] == 32


class TestBenchmarkGemm:
    """Tests for benchmark_gemm function."""

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_benchmark_gemm_basic(self):
        """Test basic benchmark_gemm functionality."""
        results = benchmark_gemm(
            sizes=[(64, 64, 64), (128, 128, 128)],
            dtype=torch.float16,
        )

        assert len(results) == 2
        for result in results:
            assert "M" in result
            assert "N" in result
            assert "K" in result
            assert "custom_ms" in result
            assert "custom_tflops" in result
            assert "reference_ms" in result
            assert "reference_tflops" in result
            assert "relative_perf" in result
            assert "bottleneck" in result

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_benchmark_gemm_fp32(self):
        """Test benchmark_gemm with FP32."""
        results = benchmark_gemm(
            sizes=[(64, 64, 64)],
            dtype=torch.float32,
        )

        assert len(results) == 1


class TestPrintBenchmarkResults:
    """Tests for print_benchmark_results function."""

    def test_print_attention_results(self, capsys):
        """Test printing attention benchmark results."""
        results = [
            {
                "seq_len": 512,
                "custom_ms": 1.0,
                "custom_tflops": 10.0,
                "reference_ms": 1.5,
                "reference_tflops": 6.7,
                "speedup": 1.5,
                "bottleneck": "memory_bound",
            }
        ]

        print_benchmark_results(results, title="Attention Benchmark")
        captured = capsys.readouterr()

        assert "Attention Benchmark" in captured.out
        assert "Speedup" in captured.out

    def test_print_gemm_results(self, capsys):
        """Test printing GEMM benchmark results."""
        results = [
            {
                "M": 1024,
                "N": 1024,
                "K": 1024,
                "custom_ms": 2.0,
                "custom_tflops": 20.0,
                "reference_ms": 1.8,
                "reference_tflops": 22.2,
                "relative_perf": 0.9,
                "bottleneck": "compute_bound",
            }
        ]

        print_benchmark_results(results, title="GEMM Benchmark")
        captured = capsys.readouterr()

        assert "GEMM Benchmark" in captured.out
        assert "Relative Performance" in captured.out

    def test_print_empty_results(self, capsys):
        """Test printing empty results."""
        print_benchmark_results([], title="Empty Results")
        captured = capsys.readouterr()

        assert "Empty Results" in captured.out
