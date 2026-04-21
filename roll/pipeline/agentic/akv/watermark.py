from dataclasses import dataclass


@dataclass
class FreeBlockWatermark:
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError("free block watermarks must be positive")
        if self.high < self.low:
            raise ValueError("free block high watermark must be greater than or equal to low watermark")

    def update(self, free_blocks: int, unloading: bool) -> bool:
        if unloading:
            return free_blocks < self.high
        return free_blocks < self.low
