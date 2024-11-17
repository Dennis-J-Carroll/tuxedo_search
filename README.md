# Tuxedo Search Engine

<div align="center">

![Tuxedo Logo](docs/images/tuxedo-logo.png)

[![Build Status](https://github.com/tuxedo-search/tuxedo/workflows/CI/badge.svg)](https://github.com/tuxedo-search/tuxedo/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

*Lightning-fast, hybrid search engine with sub-millisecond responses*
</div>

## 🚀 Overview

Tuxedo is a high-performance hybrid search engine designed for instantaneous information retrieval. By combining Python's flexibility with C's performance and Assembly optimizations, Tuxedo achieves ultra-fast search capabilities with response times ≤0.405 seconds.

### Key Features

- 🏃‍♂️ **Ultra-Fast Response**: Target response time of ≤0.405 seconds
- 🔄 **Hybrid Architecture**: Python, C, and Assembly optimizations
- 💨 **SIMD Acceleration**: AVX-512 instructions for parallel operations
- 🗄️ **Multi-Level Caching**: Progressive response system
- 🔍 **Enhanced Fuzzy Search**: Optimized approximate matching
- 🧠 **NUMA-Aware**: Optimized memory operations

## 🛠️ Requirements

- Python 3.9+
- GCC 9+ or Clang 10+
- NASM 2.15+
- CPU with AVX-512 support
- Redis 6+ (optional, for distributed caching)
- 4GB RAM minimum (8GB recommended)

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/tuxedo-search/tuxedo.git
cd tuxedo
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Build C extensions and Assembly components**
```bash
make build
```

## 🚦 Quick Start

```python
from tuxedo import SearchEngine

# Initialize the engine
engine = SearchEngine()

# Add some documents
engine.index_documents([
    {"id": "1", "content": "Python programming guide"},
    {"id": "2", "content": "Advanced search algorithms"},
    {"id": "3", "content": "Performance optimization techniques"}
])

# Perform a search
results = engine.search("python guide")

# Print results
for result in results:
    print(f"Match: {result.content}")
    print(f"Score: {result.score}")
```

## 🏗️ Architecture

Tuxedo uses a hybrid architecture combining:

- **Python**: High-level operations and API
- **C**: Core functionality and performance-critical operations
- **Assembly**: SIMD optimizations and critical path acceleration

### Core Components

1. **Query Processor**
   - Progressive response generation
   - Intelligent query routing
   - Real-time result ranking

2. **Pattern Matcher**
   - SIMD-optimized string matching
   - Approximate matching with Assembly acceleration
   - Parallel pattern processing

3. **Cache System**
   - Sub-millisecond L1 cache
   - Distributed L2 cache with Redis
   - Predictive cache warming

## 📊 Performance

| Operation          | Target Latency | Typical Latency |
|-------------------|----------------|-----------------|
| Exact Match       | ≤0.1ms        | 0.05-0.08ms    |
| Fuzzy Match       | ≤0.3ms        | 0.15-0.25ms    |
| Cache Hit (L1)    | ≤0.05ms       | 0.02-0.04ms    |
| Full Search       | ≤0.405s       | 0.2-0.3s       |

## 🔧 Configuration

Tuxedo can be configured through either environment variables or a configuration file:

```yaml
# config.yaml
engine:
  max_threads: 8
  cache_size_mb: 1024
  enable_numa: true

cache:
  l1_size_mb: 256
  l2_size_mb: 1024
  redis_url: "redis://localhost:6379"

matcher:
  use_simd: true
  fuzzy_threshold: 0.85
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details. Key areas we're currently focusing on:

- SIMD optimizations
- Cache system refinements
- Query processing improvements
- Performance monitoring

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The AVX-512 optimizations were inspired by work from Intel's optimization guides
- Fuzzy matching algorithms adapted from various academic papers
- Special thanks to all contributors and the open source community

