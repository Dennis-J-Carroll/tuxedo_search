"""Base matcher interface for all matching implementations."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseMatcher(ABC):
    """Abstract base class for all matchers."""
    
    @abstractmethod
    def search(
        self,
        query: str,
        candidates: List[str],
        score_cutoff: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Abstract search method all matchers must implement."""
        pass 