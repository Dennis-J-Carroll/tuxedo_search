# Tuxedo Search

A high-performance search engine implementation with advanced features:

## Features

- Advanced merge strategies for combining search results
- Assembly-optimized core operations for maximum performance
- Hybrid search system combining local and web results
- Comprehensive benchmarking system
- Fuzzy matching capabilities

## Components

- `advanced-merge-strategies.py`: Smart result merging with semantic analysis
- `assembly-benchmarks.py`: Performance benchmarking system
- `assembly-optimizations.py`: Low-level optimizations using assembly
- `hybrid-search-system.py`: Combined local/web search system
- `matchers/`: Fuzzy matching implementations
- `utils/`: Utility functions and performance monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tuxedo_search.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from tuxedo_search import HybridSearchSystem

# Initialize search system
search_system = HybridSearchSystem(config)

# Perform search
results = await search_system.search(
    "python programming tutorials",
    context=SearchContext(freshness_required=True)
)
```

## Performance

The system uses various optimization techniques:

- SIMD instructions for vector operations
- Assembly-optimized core algorithms
- Intelligent caching and result merging
- NUMA-aware memory management

## License

MIT License - see LICENSE file for details
