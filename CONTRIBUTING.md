# Contributing Guide

Thank you for your interest in contributing! This guide covers development setup, code standards, and contribution workflow.

---

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Build extension | `pip install -e .` |
| Run all tests | `pytest tests/ -v` |
| Run CPU-safe tests | `pytest tests/ -v -m "not cuda"` |
| Lint check | `ruff check python/ tests/ benchmarks/` |
| Format code | `ruff format python/ tests/ benchmarks/` |

---

## Development Setup

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.0+ |
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| CMake | 3.18 | 3.25+ |
| GCC/G++ | 9.0 | 11+ |
| NVIDIA GPU | SM 7.0 | SM 8.0+ |

### Setup Steps

```bash
# Clone repository
git clone https://github.com/LessUp/llm-speed.git
cd llm-speed

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest hypothesis ruff

# Build and install
pip install -e . --verbose
```

### IDE Configuration

**VS Code Extensions:**
- ms-vscode.cpptools
- ms-vscode.cmake-tools
- ms-python.python
- ms-python.pytest

---

## Code Standards

### C++/CUDA

```cpp
// Constants: UPPER_SNAKE_CASE
constexpr int BLOCK_SIZE = 256;

// Templates: PascalCase
template<typename T>

// Functions: snake_case
__global__ void attention_kernel(...)

// Variables: snake_case
int thread_idx;

// Structs: PascalCase
struct AttentionConfig { int batch_size; };
```

### Python

- PEP 8 compliance
- 4-space indentation
- 100 character max line length
- f-strings for formatting

```python
def compute_flops(batch: int, heads: int, seq_len: int) -> int:
    """Compute FLOPs for attention.
    
    Args:
        batch: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
    
    Returns:
        Total FLOPs
    """
    return batch * heads * seq_len * seq_len * 4
```

### Linting

```bash
# Check code style
ruff check python/ tests/ benchmarks/

# Auto-fix issues
ruff check --fix python/ tests/ benchmarks/

# Format code
ruff format python/ tests/ benchmarks/
```

---

## Development Workflow

### Branch Naming

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New feature | `feature/grouped-query-attention` |
| `fix/` | Bug fix | `fix/shared-memory-overflow` |
| `perf/` | Performance | `perf/register-tiling` |
| `docs/` | Documentation | `docs/api-reference` |
| `test/` | Testing | `test/boundary-cases` |

### Workflow

```bash
# Create feature branch
git checkout main
git pull origin main
git checkout -b feature/your-feature

# Make changes and test
pip install -e . --verbose
pytest tests/ -v -k "test_your_feature"

# Commit with conventional format
git add <files>
git commit -m "feat: add grouped query attention support"

# Push and create PR
git push origin feature/your-feature
```

---

## Testing

### Test Categories

| Marker | Purpose | Command |
|--------|---------|---------|
| `cuda` | Requires GPU | `pytest -m cuda` |
| `property` | Hypothesis tests | `pytest -m property` |
| `slow` | Long-running | `pytest -m "not slow"` |

### Test Structure

```python
import pytest
from hypothesis import given, settings, strategies as st

class TestFlashAttention:
    @pytest.mark.cuda
    def test_correctness(self, device):
        """Verify output matches PyTorch reference."""
        q = torch.randn(2, 4, 64, 32, device=device)
        output = flash_attention(q, k, v)
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert_close(output, reference)

    @pytest.mark.cuda
    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(batch=st.integers(1, 4), seq_len=st.integers(16, 256))
    def test_property(self, batch, seq_len, device):
        """Property-based testing with Hypothesis."""
        pass
```

### Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=python --cov-report=html
```

---

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

[optional body]
```

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat: add sliding window attention` |
| `fix` | Bug fix | `fix: resolve divide-by-zero in softmax` |
| `perf` | Performance | `perf: optimize register allocation` |
| `docs` | Documentation | `docs: add tensor core examples` |
| `test` | Testing | `test: add causal mask tests` |
| `refactor` | Refactoring | `refactor: extract reduction logic` |
| `style` | Formatting | `style: fix clang-format` |
| `chore` | Maintenance | `chore: update cuda arch flags` |

### Scopes

- `attention` — Attention kernels
- `gemm` — GEMM kernels
- `bindings` — Python bindings
- `test` — Testing
- `docs` — Documentation
- `ci` — CI/CD

---

## Pull Request Checklist

- [ ] Code passes `ruff check`
- [ ] Code passes `ruff format --check`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Commit messages follow convention
- [ ] Documentation updated if needed

---

## Issue Reporting

### Bug Report

Include:
1. Environment (OS, CUDA, Python, PyTorch, GPU)
2. Reproduction steps with code
3. Expected vs actual behavior
4. Error messages/logs

### Feature Request

Include:
1. Use case description
2. Expected benefits
3. Implementation suggestions (optional)

---

## Contact

- **Issues**: https://github.com/LessUp/llm-speed/issues
- **Discussions**: https://github.com/LessUp/llm-speed/discussions
