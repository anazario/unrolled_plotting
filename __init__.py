#!/usr/bin/env python3
"""
Unrolled Plotting Package

This package provides modular components for creating unrolled yield plots from 2D Ms vs Rs distributions.

Components:
- UnrolledDataProcessor: Data processing and yield calculation
- UnrolledHistogramMaker: Histogram creation and styling  
- UnrolledCanvasMaker: Canvas creation and plot finalization
- UnrolledPlotter: High-level interface combining all components

Usage:
    from unrolled_plotting import UnrolledPlotter
    
    plotter = UnrolledPlotter(luminosity=400.0)
    result = plotter.recreate_original_plot('signal.root', 'signal')
"""

from .unrolled_data_processor import UnrolledDataProcessor
from .unrolled_histogram_maker import UnrolledHistogramMaker
from .unrolled_canvas_maker import UnrolledCanvasMaker
from .unrolled_plotter import UnrolledPlotter

__all__ = [
    'UnrolledDataProcessor',
    'UnrolledHistogramMaker', 
    'UnrolledCanvasMaker',
    'UnrolledPlotter'
]

__version__ = '1.0.0'
__author__ = 'Generated with Claude Code'