"""plateread - license plate recovery from difficult images.

Handles the distortions that are actually invertible: perspective, rotation,
shear, uneven lighting, mild motion blur and defocus. Reports honestly when
the pixels no longer carry enough information to read.
"""

__version__ = "1.0.0"
