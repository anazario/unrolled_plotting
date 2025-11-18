#!/usr/bin/env python3
"""
UnrolledHistogramMaker - ROOT histogram creation and styling for unrolled plots.

This class handles:
- Creating ROOT histograms from 1D yield data
- Styling histograms with colors, line styles, etc.
- Creating error bands and uncertainty visualizations
- Managing histogram properties and metadata

Designed to work with UnrolledDataProcessor for modular plotting.
"""

import ROOT
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class UnrolledHistogramMaker:
    def __init__(self):
        """Initialize the histogram maker."""
        
        # Central style configuration
        self.default_line_width = 2
        self.default_line_style = 2
        self.default_fill_style = 0  # No fill
        
        # Color schemes (using ROOT predefined colors)
        self.comparison_colors = [
            ROOT.kBlue + 1,
            ROOT.kGreen + 2,
            ROOT.kRed - 4,
            ROOT.kOrange - 3,
            ROOT.kViolet - 4,
            ROOT.kBlack,
            ROOT.kAzure + 2,
            ROOT.kGray
        ]
        
        # Default styling options
        self.default_styles = {
            'signal': {
                'line_color': ROOT.kOrange+7,
                'line_width': self.default_line_width,
                'line_style': self.default_line_style,
                'fill_style': self.default_fill_style,
                'marker_style': 0
            },
            'background': {
                'line_color': ROOT.kBlue,
                'line_width': self.default_line_width,
                'line_style': self.default_line_style,
                'fill_style': self.default_fill_style,
                'marker_style': 0
            },
            'data': {
                'line_color': ROOT.kBlack,
                'line_width': self.default_line_width,
                'line_style': self.default_line_style,
                'fill_style': self.default_fill_style,
                'marker_style': 20,
                'marker_size': 1.2
            }
        }
    
    def create_histogram(self, yields_1d: np.ndarray, errors_1d: np.ndarray, 
                        bin_labels: List[str], name: str, title: str = "",
                        grouping_type: str = 'ms') -> ROOT.TH1D:
        """
        Create a ROOT histogram from unrolled 1D data.
        
        Args:
            yields_1d: 1D yield array (9 bins)
            errors_1d: 1D error array (9 bins)
            bin_labels: List of bin labels
            name: Histogram name
            title: Histogram title (empty by default)
            grouping_type: 'ms' or 'rs' for grouping type
            
        Returns:
            ROOT.TH1D histogram with data and labels
        """
        # Create histogram with 9 bins (no title)
        hist = ROOT.TH1D(name, title, 9, 0, 9)
        
        # Fill histogram with yields and errors
        for i in range(9):
            hist.SetBinContent(i + 1, yields_1d[i])
            hist.SetBinError(i + 1, errors_1d[i])
            hist.GetXaxis().SetBinLabel(i + 1, bin_labels[i])
        
        # Set axis properties with centered titles and increased spacing
        x_axis_title = "R_{S}" if grouping_type == 'ms' else "M_{S} [TeV]"
        hist.GetXaxis().SetTitle(x_axis_title)
        hist.GetXaxis().CenterTitle(True)  
        hist.GetXaxis().SetTitleOffset(1.3)
        hist.GetXaxis().SetLabelSize(0.055)
        hist.GetXaxis().SetTitleSize(0.045)
        hist.GetYaxis().SetTitle("number of events")
        hist.GetYaxis().CenterTitle(True)
        hist.GetYaxis().SetTitleSize(0.045)
        hist.GetYaxis().SetLabelSize(0.04)
        
        hist.SetStats(0)  
        
        return hist
    
    def style_histogram(self, hist: ROOT.TH1D, style_type: str = 'signal', 
                       custom_style: Optional[Dict] = None, color_index: int = 0) -> None:
        """
        Apply styling to a histogram.
        
        Args:
            hist: ROOT histogram to style
            style_type: 'signal', 'background', 'data', or 'comparison'
            custom_style: Override default styling with custom options
            color_index: Index for comparison plots (cycles through colors)
        """
        # Get style options
        if custom_style:
            style_opts = custom_style
        elif style_type == 'comparison':
            # Use cycling colors for comparison plots with centralized styling
            root_color = self.comparison_colors[color_index % len(self.comparison_colors)]
            style_opts = {
                'line_color': root_color,
                'line_width': self.default_line_width,
                'line_style': self.default_line_style,
                'fill_style': self.default_fill_style,
                'marker_style': 0
            }
        else:
            style_opts = self.default_styles.get(style_type, self.default_styles['signal'])
        
        # Apply styling
        if 'line_color' in style_opts:
            hist.SetLineColor(style_opts['line_color'])
        if 'line_width' in style_opts:
            hist.SetLineWidth(style_opts['line_width'])
        if 'line_style' in style_opts:
            hist.SetLineStyle(style_opts['line_style'])
        if 'fill_style' in style_opts:
            hist.SetFillStyle(style_opts['fill_style'])
        if 'fill_color' in style_opts:
            hist.SetFillColor(style_opts['fill_color'])
        if 'marker_style' in style_opts:
            hist.SetMarkerStyle(style_opts['marker_style'])
        if 'marker_size' in style_opts:
            hist.SetMarkerSize(style_opts['marker_size'])
        if 'marker_color' in style_opts:
            hist.SetMarkerColor(style_opts['marker_color'])
    
    def create_error_band(self, hist: ROOT.TH1D, band_style: str = 'hatched', 
                         transparency: float = 0.3) -> Optional[ROOT.TGraphAsymmErrors]:
        """
        Create an error band for a histogram.
        
        Args:
            hist: ROOT histogram to create error band for
            band_style: 'hatched', 'solid', or 'transparent'
            transparency: Transparency level (0-1) for solid bands
            
        Returns:
            ROOT.TGraphAsymmErrors for the error band, or None if no valid points
        """
        n_bins = hist.GetNbinsX()
        x_vals = []
        y_vals = []
        ex_low = []
        ex_high = []
        ey_low = []
        ey_high = []
        
        for i in range(1, n_bins + 1):
            x_center = hist.GetBinCenter(i)
            y_center = hist.GetBinContent(i)
            y_error = hist.GetBinError(i)
            bin_width = hist.GetBinWidth(i)
            
            # Skip bins with zero content to avoid issues in log scale
            if y_center <= 0:
                continue
                
            x_vals.append(x_center)
            y_vals.append(y_center)
            ex_low.append(bin_width / 2)  # Half bin width
            ex_high.append(bin_width / 2)  # Half bin width
            ey_low.append(y_error)  # Error down
            ey_high.append(y_error)  # Error up
        
        # Only create error graph if we have valid points
        if len(x_vals) == 0:
            return None
        
        # Create TGraphAsymmErrors for error band
        error_graph = ROOT.TGraphAsymmErrors(len(x_vals), 
                                           np.array(x_vals, dtype=np.float64),
                                           np.array(y_vals, dtype=np.float64),
                                           np.array(ex_low, dtype=np.float64),
                                           np.array(ex_high, dtype=np.float64),
                                           np.array(ey_low, dtype=np.float64),
                                           np.array(ey_high, dtype=np.float64))
        
        # Style the error band
        if band_style == 'hatched':
            # Use current histogram color with hatched pattern
            error_graph.SetFillColor(hist.GetLineColor() - 3)  # Lighter shade
            error_graph.SetFillStyle(3013)  # Hatched pattern
        elif band_style == 'solid':
            # Solid color with transparency
            error_graph.SetFillColor(hist.GetLineColor())
            error_graph.SetFillColorAlpha(hist.GetLineColor(), transparency)
        elif band_style == 'transparent':
            # Transparent fill
            error_graph.SetFillColor(hist.GetLineColor())
            error_graph.SetFillColorAlpha(hist.GetLineColor(), transparency)
        
        error_graph.SetLineWidth(0)  # No border
        
        return error_graph
    
    def create_ratio_histogram(self, numerator: ROOT.TH1D, denominator: ROOT.TH1D,
                              name: str, title: str = "Ratio") -> ROOT.TH1D:
        """
        Create a ratio histogram from two input histograms.
        
        Args:
            numerator: Numerator histogram
            denominator: Denominator histogram
            name: Name for ratio histogram
            title: Title for ratio histogram
            
        Returns:
            ROOT.TH1D ratio histogram
        """
        # Clone numerator to preserve binning
        ratio = numerator.Clone(name)
        ratio.SetTitle(title)
        ratio.Divide(denominator)
        
        # Style for ratio plots
        ratio.GetYaxis().SetTitle("Data/MC")
        ratio.GetYaxis().CenterTitle(True)
        ratio.GetYaxis().SetRangeUser(0.5, 1.5)  # Standard ratio range
        ratio.SetLineColor(ROOT.kBlack)
        ratio.SetMarkerStyle(20)
        ratio.SetMarkerSize(1.0)
        
        return ratio
    
    def normalize_histogram(self, hist: ROOT.TH1D, normalization: str = 'unity') -> ROOT.TH1D:
        """
        Create a normalized version of a histogram.
        
        For unrolled plots, normalizes each group (3 bins) separately by its own integral.
        
        Args:
            hist: Input histogram (9 bins = 3 groups of 3 bins each)
            normalization: 'unity' (each group area=1), 'bin_width', or 'maximum'
            
        Returns:
            Normalized histogram (clone of original)
        """
        normalized = hist.Clone(f"{hist.GetName()}_normalized")
        
        if normalization == 'unity':
            # Normalize each group separately by its own integral
            for group in range(3):  # 3 groups (Ms1, Ms2, Ms3) or (Rs1, Rs2, Rs3)
                group_start = group * 3 + 1  # ROOT bins start at 1
                group_end = group_start + 2   # 3 bins per group
                
                # Calculate integral for this group only
                group_integral = 0.0
                for bin_idx in range(group_start, group_end + 1):
                    group_integral += normalized.GetBinContent(bin_idx)
                
                # Normalize this group if it has content
                if group_integral > 0:
                    for bin_idx in range(group_start, group_end + 1):
                        old_content = normalized.GetBinContent(bin_idx)
                        old_error = normalized.GetBinError(bin_idx)
                        normalized.SetBinContent(bin_idx, old_content / group_integral)
                        normalized.SetBinError(bin_idx, old_error / group_integral)
                        
        elif normalization == 'bin_width':
            # Normalize by bin width
            normalized.Scale(1.0, "width")
        elif normalization == 'maximum':
            # Normalize each group by its maximum
            for group in range(3):
                group_start = group * 3 + 1
                group_end = group_start + 2
                
                # Find maximum in this group
                group_max = 0.0
                for bin_idx in range(group_start, group_end + 1):
                    group_max = max(group_max, normalized.GetBinContent(bin_idx))
                
                # Normalize this group by its maximum
                if group_max > 0:
                    for bin_idx in range(group_start, group_end + 1):
                        old_content = normalized.GetBinContent(bin_idx)
                        old_error = normalized.GetBinError(bin_idx)
                        normalized.SetBinContent(bin_idx, old_content / group_max)
                        normalized.SetBinError(bin_idx, old_error / group_max)
        
        # Update y-axis title
        if normalization == 'unity':
            normalized.GetYaxis().SetTitle("normalized events")
        elif normalization == 'bin_width':
            normalized.GetYaxis().SetTitle("Yield / Bin Width")
        elif normalization == 'maximum':
            normalized.GetYaxis().SetTitle("Normalized to Maximum")
        
        return normalized
    
    def set_histogram_range(self, hist: ROOT.TH1D, y_scale_factor: float = 1.5,
                           force_positive_minimum: bool = True) -> None:
        """
        Set appropriate y-axis range for histogram.
        
        Args:
            hist: Histogram to adjust
            y_scale_factor: Factor to multiply maximum by (for headroom)
            force_positive_minimum: Ensure minimum is positive for log scale
        """
        # Get current range
        current_min = hist.GetMinimum()
        current_max = hist.GetMaximum()
        
        # Set new maximum with headroom
        new_max = current_max * y_scale_factor
        hist.SetMaximum(new_max)
        
        # Handle minimum for log scale compatibility
        if force_positive_minimum and current_min <= 0:
            # Find smallest positive bin content
            min_positive = None
            for i in range(1, hist.GetNbinsX() + 1):
                content = hist.GetBinContent(i)
                if content > 0:
                    if min_positive is None or content < min_positive:
                        min_positive = content
            
            if min_positive is not None:
                hist.SetMinimum(min_positive * 0.1)  # 10% of smallest positive value
    
    def add_overflow_underflow(self, hist: ROOT.TH1D) -> None:
        """
        Add overflow and underflow bins to the histogram display.
        
        Args:
            hist: Histogram to modify
        """
        # This is more relevant for continuous distributions
        # For our binned unrolled plots, overflow/underflow should be minimal
        # But included for completeness
        
        # Get overflow content
        overflow = hist.GetBinContent(hist.GetNbinsX() + 1)
        underflow = hist.GetBinContent(0)
        
        if overflow > 0:
            # Add overflow to last bin
            last_bin = hist.GetNbinsX()
            hist.SetBinContent(last_bin, hist.GetBinContent(last_bin) + overflow)
            hist.SetBinError(last_bin, np.sqrt(hist.GetBinError(last_bin)**2 + hist.GetBinError(hist.GetNbinsX() + 1)**2))
        
        if underflow > 0:
            # Add underflow to first bin
            hist.SetBinContent(1, hist.GetBinContent(1) + underflow)
            hist.SetBinError(1, np.sqrt(hist.GetBinError(1)**2 + hist.GetBinError(0)**2))
    
    def clone_histogram_structure(self, template: ROOT.TH1D, name: str, 
                                 title: str = "") -> ROOT.TH1D:
        """
        Create a new histogram with the same binning as template but empty.
        
        Args:
            template: Template histogram for binning
            name: Name for new histogram
            title: Title for new histogram
            
        Returns:
            Empty histogram with same structure as template
        """
        # Clone and reset
        new_hist = template.Clone(name)
        new_hist.SetTitle(title)
        new_hist.Reset("ICESM")  # Reset content, errors, statistics, and metadata
        
        return new_hist
    
    def get_histogram_summary(self, hist: ROOT.TH1D) -> Dict:
        """
        Get summary statistics for a histogram.
        
        Args:
            hist: Histogram to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'name': hist.GetName(),
            'title': hist.GetTitle(),
            'entries': hist.GetEntries(),
            'integral': hist.Integral(),
            'mean': hist.GetMean(),
            'rms': hist.GetRMS(),
            'minimum': hist.GetMinimum(),
            'maximum': hist.GetMaximum(),
            'n_bins': hist.GetNbinsX()
        }
    
    def create_histogram_from_function(self, func: callable, x_range: Tuple[float, float],
                                     n_bins: int, name: str, title: str = "") -> ROOT.TH1D:
        """
        Create histogram by evaluating a function (for systematic variations, etc.).
        
        Args:
            func: Function that takes x and returns y
            x_range: (min, max) for x-axis
            n_bins: Number of bins
            name: Histogram name
            title: Histogram title
            
        Returns:
            ROOT.TH1D with function values
        """
        hist = ROOT.TH1D(name, title, n_bins, x_range[0], x_range[1])
        
        for i in range(1, n_bins + 1):
            x_center = hist.GetBinCenter(i)
            y_value = func(x_center)
            hist.SetBinContent(i, y_value)
        
        return hist
