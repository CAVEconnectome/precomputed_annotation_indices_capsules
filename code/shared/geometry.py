"""
Minimal geometry classes extracted from zetta_utils for the precomp_anno library.
Contains Vec3D and BBox3D with essential functionality only.
"""

import math
from typing import Sequence, Tuple, Union

VEC3D_PRECISION = 10


class Vec3D:
    """3-dimensional vector with basic operations."""
    
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y) 
        self.z = float(z)
    
    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Vec3D index out of range")
    
    def __len__(self) -> int:
        return 3
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
    
    def __mul__(self, other: Union['Vec3D', float]) -> 'Vec3D':
        if isinstance(other, Vec3D):
            return Vec3D(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vec3D(self.x * other, self.y * other, self.z * other)
    
    def __truediv__(self, other: Union['Vec3D', float]) -> 'Vec3D':
        if isinstance(other, Vec3D):
            return Vec3D(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return Vec3D(self.x / other, self.y / other, self.z / other)
    
    def __sub__(self, other: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __add__(self, other: 'Vec3D') -> 'Vec3D':
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __repr__(self) -> str:
        return f"Vec3D({self.x}, {self.y}, {self.z})"


class BBox3D:
    """3-dimensional axis-aligned bounding box."""
    
    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
        self.bounds = tuple(
            (round(start, VEC3D_PRECISION), round(end, VEC3D_PRECISION))
            for start, end in bounds
        )
    
    @classmethod
    def from_coords(
        cls, 
        lower_bound: Sequence[float], 
        upper_bound: Sequence[float],
        resolution: Sequence[float] = (1, 1, 1)
    ) -> 'BBox3D':
        """Create BBox3D from coordinate arrays."""
        bounds = tuple(
            (lower_bound[i] * resolution[i], upper_bound[i] * resolution[i])
            for i in range(3)
        )
        return cls(bounds)
    
    @property
    def start(self) -> Vec3D:
        return Vec3D(*(b[0] for b in self.bounds))
    
    @property
    def end(self) -> Vec3D:
        return Vec3D(*(b[1] for b in self.bounds))
    
    @property
    def shape(self) -> Vec3D:
        return self.end - self.start
    
    def contains(self, point: Sequence[float], resolution: Sequence[float] = (1, 1, 1)) -> bool:
        """Check if point is within the bounding box."""
        for i in range(3):
            coord = point[i] * resolution[i]
            if not (self.bounds[i][0] <= coord <= self.bounds[i][1]):
                return False
        return True
    
    def line_intersects(self, start: Sequence[float], end: Sequence[float], resolution: Sequence[float] = (1, 1, 1)) -> bool:
        """Check if line segment intersects the bounding box using separating axis theorem."""
        # Convert to actual coordinates
        line_start = Vec3D(*(start[i] * resolution[i] for i in range(3)))
        line_end = Vec3D(*(end[i] * resolution[i] for i in range(3)))
        
        # Check if either endpoint is inside the box
        if self.contains(start, resolution) or self.contains(end, resolution):
            return True
        
        # Use parametric line representation: P(t) = start + t * (end - start) for t in [0,1]
        direction = line_end - line_start
        
        # Check intersection with each axis-aligned plane
        t_min = 0.0
        t_max = 1.0
        
        for i in range(3):
            if abs(direction[i]) < 1e-9:  # Line is parallel to this axis
                # Check if line is outside the slab
                if line_start[i] < self.bounds[i][0] or line_start[i] > self.bounds[i][1]:
                    return False
            else:
                # Calculate intersection parameters with the two planes
                t1 = (self.bounds[i][0] - line_start[i]) / direction[i]
                t2 = (self.bounds[i][1] - line_start[i]) / direction[i]
                
                # Ensure t1 <= t2
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # Update the intersection interval
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                
                # If t_min > t_max, no intersection
                if t_min > t_max:
                    return False
        
        return True
    
    def __repr__(self) -> str:
        return f"BBox3D({self.bounds})"


def round(value: float, precision: int = VEC3D_PRECISION) -> float:
    """Round to specified decimal places."""
    return __builtins__['round'](value, precision)