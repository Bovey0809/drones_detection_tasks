"""Configuration module for drone detection."""

from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class DetectionConfig:
    """Configuration for drone detection parameters."""
    
    confidence_threshold: float = 0.5
    min_size: int = 50
    max_size: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DetectionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.min_size <= 0:
            raise ValueError("min_size must be positive")
        
        if self.max_size <= self.min_size:
            raise ValueError("max_size must be greater than min_size")
        
        return True
