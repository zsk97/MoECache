from dataclasses import dataclass

@dataclass
class CacheConfig:
    cache_size: int
    num_layers: int
    evict_policy: str