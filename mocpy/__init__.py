# Licensed under a 3-clause BSD style license - see LICENSE
"""
MOCPy is a Python library allowing easy creation and manipulation of MOCs (Multi-Order Coverage maps).
MOC is an `IVOA standard <http://ivoa.net/documents/MOC/>`__ enabling description of arbitrary sky regions.
Based on the HEALPix sky tessellation, it maps regions on the sky into hierarchically grouped predefined cells.
"""

from .moc import MOC
from .version import __version__

__all__ = [
    'MOC',
]
