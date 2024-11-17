"""
Advanced Hybrid Search System with Smart Source Selection
"""
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum
import time
import logging

class SearchSource(Enum):
    LOCAL = "local"
    WEB = "web"
    HYBRID = "hybrid"

@dataclass
class SearchContext:
    """Context for search decision making"""
    query: str
    user_location: Optional[str] = None
    time_constraint_ms: Optional[int] = None
    freshness_required: bool = False
    deep_search: bool = False

@dataclass
class SearchMetrics:
    """Metrics for search performance tracking"""
    response_time_ms: float
    source_used: SearchSource
    cache_hit: bool
    result_count: int
    relevance_score: float

class HybridSearchSystem:
    """
    Advanced hybrid search system that intelligently combines local and web search
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._init_subsystems()
        
    def _init_subsystems(self):
        """Initialize all search subsystems"""
        # Local search system
        self.local_search = LocalSearchEngine(
            index_path=self.config['index_path'],
            cache_size=self.config['cache_size']
        )
        
        # Web search integration
        self.web_search = WebSearchIntegration(
            api_keys=self.config['api_keys'],
            rate_limits=self.config['rate_limits']
        )
        
        # Caching system
        self.cache = MultiLevelCache(
            l1_size=1000,  # Hot queries
            l2_size=10000, # Warm queries
            l3_size=100000 # Cold queries
        )
        
        # Decision engine
        self.decision_engine = SearchDecisionEngine(
            model_path=self.config['decision_model']
        )
    
    async def search(self, query: str, context: SearchContext) -> Dict:
        """
        Main search method with smart source selection
        """
        start_time = time.perf_counter()
        
        # 1. Quick cache check
        if cached_result := self.cache.get(query):
            if self._is_cache_valid(cached_result, context):
                return self._enhance_cached_result(cached_result, context)
        
        # 2. Determine search strategy
        strategy = self.decision_engine.determine_strategy(query, context)
        
        # 3. Execute search based on strategy
        results = await self._execute_search_strategy(strategy, query, context)
        
        # 4. Track metrics
        self._track_metrics(SearchMetrics(
            response_time_ms=(time.perf_counter() - start_time) * 1000,
            source_used=strategy.source,
            cache_hit=bool(cached_result),
            result_count=len(results),
            relevance_score=self._calculate_relevance(results)
        ))
        
        return results
    
    async def _execute_search_strategy(self, 
                                     strategy: 'SearchStrategy', 
                                     query: str, 
                                     context: SearchContext) -> Dict:
        """Execute the determined search strategy"""
        if strategy.source == SearchSource.LOCAL:
            return await self._local_only_search(query, context)
        elif strategy.source == SearchSource.WEB:
            return await self._web_only_search(query, context)
        else:
            return await self._hybrid_search(query, context, strategy)
    
    async def _hybrid_search(self, 
                           query: str, 
                           context: SearchContext, 
                           strategy: 'SearchStrategy') -> Dict:
        """
        Perform hybrid search with intelligent merging
        """
        # 1. Launch parallel searches
        local_task = asyncio.create_task(
            self._local_only_search(query, context)
        )
        web_task = asyncio.create_task(
            self._web_only_search(query, context)
        )
        
        # 2. Wait for results with timeout
        try:
            results = await asyncio.gather(
                local_task, 
                web_task,
                return_exceptions=True
            )
        except asyncio.TimeoutError:
            # Handle timeout gracefully
            results = await self._handle_timeout(local_task, web_task)
        
        # 3. Merge results intelligently
        merged_results = self._smart_merge(
            local_results=results[0] if not isinstance(results[0], Exception) else None,
            web_results=results[1] if not isinstance(results[1], Exception) else None,
            strategy=strategy
        )
        
        # 4. Cache results
        self.cache.set(query, merged_results)
        
        return merged_results
    
    def _smart_merge(self, 
                    local_results: Optional[Dict], 
                    web_results: Optional[Dict], 
                    strategy: 'SearchStrategy') -> Dict:
        """
        Intelligently merge results based on various factors
        """
        merged = []
        local_weight = strategy.local_weight
        web_weight = strategy.web_weight
        
        # Handle cases where one source failed
        if not local_results:
            return web_results
        if not web_results:
            return local_results
        
        # Scoring and ranking
        scored_results = []
        
        # Score local results
        for result in local_results['items']:
            score = self._calculate_result_score(
                result, 
                source='local',
                weight=local_weight
            )
            scored_results.append((score, 'local', result))
        
        # Score web results
        for result in web_results['items']:
            score = self._calculate_result_score(
                result,
                source='web',
                weight=web_weight
            )
            scored_results.append((score, 'web', result))
        
        # Sort by score and deduplicate
        scored_results.sort(reverse=True)
        seen_urls = set()
        
        for score, source, result in scored_results:
            url = result.get('url')
            if url not in seen_urls:
                merged.append(result)
                seen_urls.add(url)
        
        return {
            'items': merged,
            'total_count': len(merged),
            'sources_used': {
                'local': bool(local_results),
                'web': bool(web_results)
            }
        }
    
    def _calculate_result_score(self, 
                              result: Dict, 
                              source: str, 
                              weight: float) -> float:
        """
        Calculate a score for a single result
        """
        base_score = 0.0
        
        # Relevance factors
        if 'relevance_score' in result:
            base_score += result['relevance_score'] * 0.4
        
        # Freshness
        if 'timestamp' in result:
            freshness_score = self._calculate_freshness_score(result['timestamp'])
            base_score += freshness_score * 0.2
        
        # Source-specific factors
        if source == 'local':
            # Local results might have additional metrics
            if 'popularity' in result:
                base_score += result['popularity'] * 0.2
        else:
            # Web results might have PageRank-like metrics
            if 'page_rank' in result:
                base_score += result['page_rank'] * 0.2
        
        # Apply source weight
        return base_score * weight
    
    async def _handle_timeout(self, 
                            local_task: asyncio.Task, 
                            web_task: asyncio.Task) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Handle timeout situations gracefully
        """
        results = [None, None]
        
        # Try to get what we can from each task
        for i, task in enumerate([local_task, web_task]):
            try:
                if not task.done():
                    # Give it a very short additional time
                    results[i] = await asyncio.wait_for(task, timeout=0.1)
                else:
                    results[i] = task.result()
            except Exception as e:
                self.logger.error(f"Error in search task: {e}")
                results[i] = None
        
        return tuple(results)

class SearchDecisionEngine:
    """
    Engine for making intelligent decisions about search strategy
    """
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.query_analyzer = QueryAnalyzer()
    
    def determine_strategy(self, 
                         query: str, 
                         context: SearchContext) -> 'SearchStrategy':
        """
        Determine the optimal search strategy based on various factors
        """
        # 1. Analyze query
        query_features = self.query_analyzer.analyze(query)
        
        # 2. Consider context
        time_sensitive = context.time_constraint_ms is not None
        freshness_needed = context.freshness_required
        
        # 3. Make decision
        if time_sensitive and context.time_constraint_ms < 50:
            # Very tight time constraint - use local only
            return SearchStrategy(
                source=SearchSource.LOCAL,
                local_weight=1.0,
                web_weight=0.0
            )
        
        if freshness_needed:
            # Need fresh results - prioritize web
            return SearchStrategy(
                source=SearchSource.HYBRID,
                local_weight=0.3,
                web_weight=0.7
            )
        
        # Use model for other cases
        return self._model_based_decision(query_features, context)

@dataclass
class SearchStrategy:
    """
    Search strategy configuration
    """
    source: SearchSource
    local_weight: float = 0.5
    web_weight: float = 0.5
    timeout_ms: int = 1000
    max_results: int = 20

# Usage example
async def main():
    # Configuration
    config = {
        'index_path': './search_index',
        'cache_size': 10000,
        'api_keys': {
            'google': 'your_key',
            'bing': 'your_key'
        },
        'rate_limits': {
            'queries_per_second': 10
        },
        'decision_model': './models/decision.pkl'
    }
    
    # Initialize system
    search_system = HybridSearchSystem(config)
    
    # Example search
    context = SearchContext(
        query="python programming",
        freshness_required=True,
        time_constraint_ms=100
    )
    
    results = await search_system.search(
        "python programming tutorials",
        context
    )
    
    print(f"Found {results['total_count']} results")
    print(f"Sources used: {results['sources_used']}")

if __name__ == "__main__":
    asyncio.run(main())