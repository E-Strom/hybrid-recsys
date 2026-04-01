
from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    @abstractmethod
    def recommend(self, user_id: int, K: int) -> List[int]:
        """
        Return top-K item IDs for a given user.
        """
        pass

    def recommend_batch(self, user_ids: List[int], K: int) -> List[List[int]]:
        """
        Optional: default fallback (slow)
        """
        return [self.recommend(u, K) for u in user_ids]