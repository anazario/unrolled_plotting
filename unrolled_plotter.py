#!/usr/bin/env python3
"""
UnrolledPlotter - Main interface for creating unrolled yield plots.

This class provides a high-level interface that combines:
- UnrolledDataProcessor: Data processing and yield calculation
- UnrolledHistogramMaker: Histogram creation and styling
- UnrolledCanvasMaker: Canvas creation and plot finalization

Designed to reproduce the exact functionality of unrolled_yield_analysis.py
with improved modularity for future extensions.
"""

import ROOT
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from unrolled_data_processor import UnrolledDataProcessor
from unrolled_histogram_maker import UnrolledHistogramMaker  
from unrolled_canvas_maker import UnrolledCanvasMaker

# Enable batch mode
ROOT.gROOT.SetBatch(True)

class UnrolledPlotter:
    def __init__(self, luminosity: float = 400.0, grouping_type: str = 'ms',
                 baseline_cuts: List[str] = None):
        """
        Initialize the unrolled plotter.
        
        Args:
            luminosity: Integrated luminosity in fb-1 (default 400)
            grouping_type: 'ms' or 'rs' for grouping type (default 'ms')
            baseline_cuts: List of baseline cuts (default: ['selCMet>150', 'hlt_flags', 'cleaning_flags'])
        """
        self.luminosity = luminosity
        self.grouping_type = grouping_type
        
        # Default baseline cuts
        if baseline_cuts is None:
            baseline_cuts = ['selCMet>150', 'hlt_flags', 'cleaning_flags', 'rjrCleaningVeto0']
        self.baseline_cuts = baseline_cuts
        
        # Initialize components
        self.data_processor = UnrolledDataProcessor(luminosity)
        self.histogram_maker = UnrolledHistogramMaker()
        self.canvas_maker = UnrolledCanvasMaker(luminosity)
        
        # Analysis tracking
        self.analysis_summary = {
            'plots_created': [],
            'files_processed': [],
            'errors': []
        }
    
    def create_single_plot(self, file_path: str, file_type: str, 
                          final_state_flags: List[str] = None,
                          signal_scale: float = 1.0, background_scale: float = 1.0,
                          name_suffix: str = "", style_type: str = None,
                          output_path: str = None, 
                          output_formats: List[str] = ['pdf'],
                          show_error_band: bool = True) -> Dict:
        """
        Create a single unrolled plot from one file.
        
        Args:
            file_path: Path to ROOT file
            file_type: 'signal', 'background', or 'data'
            final_state_flags: List of final state selection flags
            signal_scale: Scaling factor for signal
            background_scale: Scaling factor for background  
            name_suffix: Suffix to add to histogram/canvas names
            style_type: Override automatic style selection
            output_path: Output file path (without extension)
            output_formats: List of output formats
            show_error_band: Whether to show error band
            
        Returns:
            Dictionary with histogram, canvas, and metadata
        """
        try:
            # Step 1: Load file data once
            file_data = self.data_processor.load_file_data(
                file_path, file_type, self.baseline_cuts, signal_scale, background_scale
            )
            
            # Step 2: Calculate 2D yields from cached data
            yields_2d, errors_2d = self.data_processor.calculate_2d_yields_from_data(
                file_data, final_state_flags
            )
            
            # Step 3: Unroll to 1D
            yields_1d, errors_1d, bin_labels = self.data_processor.unroll_2d_to_1d(
                yields_2d, errors_2d, self.grouping_type
            )
            
            # Step 4: Create histogram
            file_basename = file_path.split('/')[-1].split('.')[0]
            hist_name = f"{file_basename}_{self.grouping_type}_{file_type}{name_suffix}"
            
            hist = self.histogram_maker.create_histogram(
                yields_1d, errors_1d, bin_labels, hist_name, "", self.grouping_type
            )
            
            # Step 5: Style histogram
            if style_type is None:
                style_type = file_type
            self.histogram_maker.style_histogram(hist, style_type)
            
            # Step 6: Set proper y-axis range (5x for log scale like current implementation)
            self.histogram_maker.set_histogram_range(hist, y_scale_factor=5.0)
            
            # Step 7: Create error band if requested
            error_band = None
            if show_error_band:
                error_band = self.histogram_maker.create_error_band(hist, 'hatched')
            
            # Step 8: Create canvas
            canvas_name = f"{hist_name}_canvas"
            canvas = self.canvas_maker.create_base_canvas(canvas_name, "", 
                                                        use_log_y=True, use_grid=True)
            
            # Step 9: Get group labels
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            
            # Step 10: Finalize canvas with all decorations
            self.canvas_maker.finalize_canvas(canvas, hist, group_labels, error_band)
            
            # Step 11: Save if output path provided
            if output_path:
                self.canvas_maker.save_canvas(canvas, output_path, output_formats)
                self.analysis_summary['plots_created'].append(f"{output_path} ({', '.join(output_formats)})")
            
            # Track processing
            self.analysis_summary['files_processed'].append(f"{file_type}: {file_path}")
            
            return {
                'success': True,
                'histogram': hist,
                'canvas': canvas,
                'error_band': error_band,
                'yields_2d': yields_2d,
                'errors_2d': errors_2d,
                'yields_1d': yields_1d,
                'errors_1d': errors_1d,
                'bin_labels': bin_labels,
                'group_labels': group_labels,
                'file_type': file_type,
                'total_yield': np.sum(yields_1d),
                'n_bins_with_data': np.sum(yields_1d > 0)
            }
            
        except Exception as e:
            error_msg = f"Error creating plot for {file_path}: {str(e)}"
            self.analysis_summary['errors'].append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_path': file_path,
                'file_type': file_type
            }
    
    def create_comparison_plot(self, file_configs: List[Dict], 
                              final_state_flags: List[str] = None,
                              labels: List[str] = None,
                              name: str = "comparison",
                              output_path: str = None,
                              output_formats: List[str] = ['pdf'],
                              normalize: bool = False) -> Dict:
        """
        Create a comparison plot with multiple curves.
        
        Args:
            file_configs: List of dicts with 'path', 'type', 'scale', 'label' keys
            final_state_flags: Final state selection flags
            labels: Legend labels (auto-generated if None)
            name: Base name for plot
            output_path: Output file path
            output_formats: Output formats
            normalize: Whether to normalize histograms
            
        Returns:
            Dictionary with canvas and metadata
        """
        if not file_configs:
            return {'success': False, 'error': 'No file configurations provided'}
        
        try:
            histograms = []
            plot_labels = []
            
            for i, config in enumerate(file_configs):
                # Create individual plot
                result = self.create_single_plot(
                    config['path'], config['type'],
                    final_state_flags=final_state_flags,
                    signal_scale=config.get('scale', 1.0),
                    style_type='comparison'
                )
                
                if result['success']:
                    hist = result['histogram']
                    
                    # Apply comparison styling with color index
                    self.histogram_maker.style_histogram(hist, 'comparison', color_index=i)
                    
                    # Normalize if requested
                    if normalize:
                        hist = self.histogram_maker.normalize_histogram(hist, 'unity')
                    
                    histograms.append(hist)
                    
                    # Generate label
                    if labels and i < len(labels):
                        plot_labels.append(labels[i])
                    else:
                        plot_labels.append(config.get('label', f"{config['type']} {i+1}"))
                else:
                    self.analysis_summary['errors'].append(result['error'])
            
            if not histograms:
                return {'success': False, 'error': 'No valid histograms created'}
            
            # Create comparison canvas
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            canvas = self.canvas_maker.create_comparison_canvas(
                histograms, plot_labels, group_labels, name
            )
            
            # Save if output path provided
            if output_path:
                self.canvas_maker.save_canvas(canvas, output_path, output_formats)
                self.analysis_summary['plots_created'].append(f"{output_path} comparison ({', '.join(output_formats)})")
            
            return {
                'success': True,
                'canvas': canvas,
                'histograms': histograms,
                'labels': plot_labels,
                'group_labels': group_labels,
                'normalized': normalize
            }
            
        except Exception as e:
            error_msg = f"Error creating comparison plot: {str(e)}"
            self.analysis_summary['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}
    
    def create_multi_final_state_plot(self, file_path: str, file_type: str,
                                     final_state_flags: List[str],
                                     labels: List[str] = None,
                                     signal_scale: float = 1.0,
                                     background_scale: float = 1.0,
                                     name: str = "multi_final_states",
                                     output_path: str = None,
                                     output_formats: List[str] = ['pdf'],
                                     normalize: bool = False) -> Dict:
        """
        Create a plot with multiple final states using cached file data.
        
        This is much more efficient than create_comparison_plot() for multiple
        final states from the same file since it loads the file only once.
        
        Args:
            file_path: Path to ROOT file
            file_type: 'signal', 'background', or 'data'
            final_state_flags: List of final state selection flags
            labels: Legend labels (auto-generated if None)
            signal_scale: Signal scaling factor
            background_scale: Background scaling factor
            name: Base name for plot
            output_path: Output file path
            output_formats: Output formats
            
        Returns:
            Dictionary with canvas and metadata
        """
        if not final_state_flags:
            return {'success': False, 'error': 'No final state flags provided'}
        
        try:
            # Step 1: Load file data once (applies baseline cuts)
            file_data = self.data_processor.load_file_data(
                file_path, file_type, self.baseline_cuts, signal_scale, background_scale
            )
            
            # Step 2: Create histograms for each final state
            histograms = []
            plot_labels = []
            
            for i, final_state in enumerate(final_state_flags):
                # Calculate yields for this final state
                yields_2d, errors_2d = self.data_processor.calculate_2d_yields_from_data(
                    file_data, [final_state]
                )
                
                # Unroll to 1D
                yields_1d, errors_1d, bin_labels = self.data_processor.unroll_2d_to_1d(
                    yields_2d, errors_2d, self.grouping_type
                )
                
                # Create histogram
                hist_name = f"{name}_{final_state}_{i}"
                hist = self.histogram_maker.create_histogram(
                    yields_1d, errors_1d, bin_labels, hist_name, "", self.grouping_type
                )
                
                # Apply comparison styling with color index
                self.histogram_maker.style_histogram(hist, 'comparison', color_index=i)
                
                # Normalize if requested
                if normalize:
                    hist = self.histogram_maker.normalize_histogram(hist, 'unity')
                
                histograms.append(hist)
                
                # Generate label
                if labels and i < len(labels):
                    plot_labels.append(labels[i])
                else:
                    # Convert to SV notation: {N}SV_{flavor}^{selection}
                    clean_label = self.canvas_maker._format_sv_label(final_state)
                    plot_labels.append(clean_label)
            
            # Step 3: Create comparison canvas
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            canvas = self.canvas_maker.create_comparison_canvas(
                histograms, plot_labels, group_labels, name
            )
            
            # Step 4: Save if output path provided
            if output_path:
                self.canvas_maker.save_canvas(canvas, output_path, output_formats)
                self.analysis_summary['plots_created'].append(f"{output_path} multi-final-states ({', '.join(output_formats)})")
            
            # Track processing
            self.analysis_summary['files_processed'].append(f"Multi-final-states from {file_type}: {file_path}")
            
            return {
                'success': True,
                'canvas': canvas,
                'histograms': histograms,
                'labels': plot_labels,
                'group_labels': group_labels,
                'file_data': file_data,
                'final_states': final_state_flags
            }
            
        except Exception as e:
            error_msg = f"Error creating multi-final-state plot: {str(e)}"
            self.analysis_summary['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}
    
    def create_comparison_plot_with_markers(self, file_configs: List[Dict], 
                                          final_state_flags: List[str] = None,
                                          labels: List[str] = None,
                                          name: str = "comparison",
                                          output_path: str = None,
                                          output_formats: List[str] = ['pdf'],
                                          normalize: bool = False,
                                          marker_styles: List[int] = None) -> Dict:
        """
        Create a comparison plot with multiple curves using offset markers for jittered visualization.
        
        Args:
            file_configs: List of dicts with 'path', 'type', 'scale', 'label' keys
            final_state_flags: Final state selection flags
            labels: Legend labels (auto-generated if None)
            name: Base name for plot
            output_path: Output file path
            output_formats: Output formats
            normalize: Whether to normalize histograms
            marker_styles: Custom marker styles for each histogram
            
        Returns:
            Dictionary with canvas and metadata
        """
        if not file_configs:
            return {'success': False, 'error': 'No file configurations provided'}
        
        try:
            histograms = []
            plot_labels = []
            original_yields = []  # Store original yields before normalization
            
            for i, config in enumerate(file_configs):
                # Create individual plot
                result = self.create_single_plot(
                    config['path'], config['type'],
                    final_state_flags=final_state_flags,
                    signal_scale=config.get('scale', 1.0),
                    style_type='comparison'
                )
                
                if result['success']:
                    hist = result['histogram']
                    
                    # Apply comparison styling with color index
                    self.histogram_maker.style_histogram(hist, 'comparison', color_index=i)
                    
                    # Store original yield before normalization
                    original_yields.append(hist.Integral())
                    
                    # Normalize if requested
                    if normalize:
                        hist = self.histogram_maker.normalize_histogram(hist, 'unity')
                    
                    histograms.append(hist)
                    
                    # Generate label
                    if labels and i < len(labels):
                        plot_labels.append(labels[i])
                    else:
                        plot_labels.append(config.get('label', f"{config['type']} {i+1}"))
                else:
                    self.analysis_summary['errors'].append(result['error'])
            
            if not histograms:
                return {'success': False, 'error': 'No valid histograms created'}
            
            # Create comparison canvas with markers
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            canvas = self.canvas_maker.create_comparison_canvas_with_markers(
                histograms, plot_labels, group_labels, name, marker_styles, original_yields=original_yields
            )
            
            # Save if output path provided
            if output_path:
                self.canvas_maker.save_canvas(canvas, output_path, output_formats)
                self.analysis_summary['plots_created'].append(f"{output_path} comparison_markers ({', '.join(output_formats)})")
            
            return {
                'success': True,
                'canvas': canvas,
                'histograms': histograms,
                'labels': plot_labels,
                'group_labels': group_labels,
                'normalized': normalize,
                'plot_type': 'markers'
            }
            
        except Exception as e:
            error_msg = f"Error creating comparison plot with markers: {str(e)}"
            self.analysis_summary['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}
    
    
    def create_plot_from_cached_data(self, file_data, final_state_flags: List[str] = None,
                                   name_suffix: str = "", style_type: str = None,
                                   show_error_band: bool = True) -> Dict:
        """
        Create a single plot from pre-loaded FileData.
        
        Args:
            file_data: Pre-loaded FileData object
            final_state_flags: Final state selection flags
            name_suffix: Suffix to add to histogram/canvas names
            style_type: Override automatic style selection
            show_error_band: Whether to show error band
            
        Returns:
            Dictionary with histogram, canvas, and metadata
        """
        try:
            # Calculate 2D yields from cached data
            yields_2d, errors_2d = self.data_processor.calculate_2d_yields_from_data(
                file_data, final_state_flags
            )
            
            # Unroll to 1D
            yields_1d, errors_1d, bin_labels = self.data_processor.unroll_2d_to_1d(
                yields_2d, errors_2d, self.grouping_type
            )
            
            # Create histogram
            file_basename = file_data.file_path.split('/')[-1].split('.')[0]
            hist_name = f"{file_basename}_{self.grouping_type}_{file_data.file_type}{name_suffix}"
            
            hist = self.histogram_maker.create_histogram(
                yields_1d, errors_1d, bin_labels, hist_name, "", self.grouping_type
            )
            
            # Style histogram
            if style_type is None:
                style_type = file_data.file_type
            self.histogram_maker.style_histogram(hist, style_type)
            
            # Set proper y-axis range
            self.histogram_maker.set_histogram_range(hist, y_scale_factor=5.0)
            
            # Create error band if requested
            error_band = None
            if show_error_band:
                error_band = self.histogram_maker.create_error_band(hist, 'hatched')
            
            # Create canvas
            canvas_name = f"{hist_name}_canvas"
            canvas = self.canvas_maker.create_base_canvas(canvas_name, "", 
                                                        use_log_y=True, use_grid=True)
            
            # Get group labels and finalize canvas
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            self.canvas_maker.finalize_canvas(canvas, hist, group_labels, error_band)
            
            return {
                'success': True,
                'histogram': hist,
                'canvas': canvas,
                'error_band': error_band,
                'yields_2d': yields_2d,
                'errors_2d': errors_2d,
                'yields_1d': yields_1d,
                'errors_1d': errors_1d,
                'bin_labels': bin_labels,
                'group_labels': group_labels,
                'file_type': file_data.file_type,
                'total_yield': np.sum(yields_1d),
                'n_bins_with_data': np.sum(yields_1d > 0)
            }
            
        except Exception as e:
            error_msg = f"Error creating plot from cached data: {str(e)}"
            self.analysis_summary['errors'].append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_path': file_data.file_path,
                'file_type': file_data.file_type
            }
    
    def create_multi_final_state_plot_with_markers(self, file_path: str, file_type: str,
                                                 final_state_flags: List[str],
                                                 labels: List[str] = None,
                                                 signal_scale: float = 1.0,
                                                 background_scale: float = 1.0,
                                                 name: str = "multi_final_states",
                                                 output_path: str = None,
                                                 output_formats: List[str] = ['pdf'],
                                                 normalize: bool = False,
                                                 marker_styles: List[int] = None) -> Dict:
        """
        Create a plot with multiple final states using offset markers for jittered visualization.
        
        Args:
            file_path: Path to ROOT file
            file_type: 'signal', 'background', or 'data'
            final_state_flags: List of final state selection flags
            labels: Legend labels (auto-generated if None)
            signal_scale: Signal scaling factor
            background_scale: Background scaling factor
            name: Base name for plot
            output_path: Output file path
            output_formats: Output formats
            normalize: Whether to normalize histograms
            marker_styles: Custom marker styles for each histogram
            
        Returns:
            Dictionary with canvas and metadata
        """
        if not final_state_flags:
            return {'success': False, 'error': 'No final state flags provided'}
        
        try:
            # Step 1: Load file data once (applies baseline cuts)
            file_data = self.data_processor.load_file_data(
                file_path, file_type, self.baseline_cuts, signal_scale, background_scale
            )
            
            # Step 2: Create histograms for each final state
            histograms = []
            plot_labels = []
            
            for i, final_state in enumerate(final_state_flags):
                # Calculate yields for this final state
                yields_2d, errors_2d = self.data_processor.calculate_2d_yields_from_data(
                    file_data, [final_state]
                )
                
                # Unroll to 1D
                yields_1d, errors_1d, bin_labels = self.data_processor.unroll_2d_to_1d(
                    yields_2d, errors_2d, self.grouping_type
                )
                
                # Create histogram
                hist_name = f"{name}_{final_state}_{i}"
                hist = self.histogram_maker.create_histogram(
                    yields_1d, errors_1d, bin_labels, hist_name, "", self.grouping_type
                )
                
                # Apply comparison styling with color index
                self.histogram_maker.style_histogram(hist, 'comparison', color_index=i)
                
                # Normalize if requested
                if normalize:
                    hist = self.histogram_maker.normalize_histogram(hist, 'unity')
                
                histograms.append(hist)
                
                # Generate label
                if labels and i < len(labels):
                    plot_labels.append(labels[i])
                else:
                    # Convert to SV notation: {N}SV_{flavor}^{selection}
                    clean_label = self.canvas_maker._format_sv_label(final_state)
                    plot_labels.append(clean_label)
            
            # Step 3: Create comparison canvas with markers
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            canvas = self.canvas_maker.create_comparison_canvas_with_markers(
                histograms, plot_labels, group_labels, name, marker_styles
            )
            
            # Step 4: Save if output path provided
            if output_path:
                self.canvas_maker.save_canvas(canvas, output_path, output_formats)
                self.analysis_summary['plots_created'].append(f"{output_path} multi-final-states-markers ({', '.join(output_formats)})")
            
            # Track processing
            self.analysis_summary['files_processed'].append(f"Multi-final-states-markers from {file_type}: {file_path}")
            
            return {
                'success': True,
                'canvas': canvas,
                'histograms': histograms,
                'labels': plot_labels,
                'group_labels': group_labels,
                'file_data': file_data,
                'final_states': final_state_flags,
                'plot_type': 'markers'
            }
            
        except Exception as e:
            error_msg = f"Error creating multi-final-state plot with markers: {str(e)}"
            self.analysis_summary['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}
    
    def create_datamc_ratio_plot(self, data_file: str, mc_files: List[str], 
                                final_state_flags: List[str] = None,
                                mc_labels: List[str] = None,
                                data_scale: float = 1.0,
                                mc_scales: List[float] = None,
                                name: str = "datamc_ratio",
                                output_path: str = None,
                                output_formats: List[str] = ['pdf'],
                                normalize: bool = False) -> Dict:
        """
        Create Data/MC ratio plot with stacked backgrounds for a single final state.
        
        Args:
            data_file: Path to data ROOT file
            mc_files: List of MC background ROOT files
            final_state_flags: Final state selection flags (only first one used)
            mc_labels: Labels for MC backgrounds (auto-generated if None)
            data_scale: Data scaling factor
            mc_scales: MC scaling factors for each file
            name: Base name for plot
            output_path: Output file path
            output_formats: Output formats
            
        Returns:
            Dictionary with canvas and metadata
        """
        # Use only the first final state if multiple provided
        final_state = final_state_flags[0] if final_state_flags else None
        return self._create_single_datamc_ratio_plot(
            data_file, mc_files, final_state, mc_labels, data_scale, 
            mc_scales, name, output_path, output_formats
        )
    
    def create_multi_datamc_ratio_plots(self, data_file: str, mc_files: List[str], 
                                      final_state_flags: List[str],
                                      mc_labels: List[str] = None,
                                      data_scale: float = 1.0,
                                      mc_scales: List[float] = None,
                                      base_name: str = "datamc_ratio",
                                      output_path: str = None,
                                      output_formats: List[str] = ['pdf'],
                                      create_both_groupings: bool = True,
                                      normalize: bool = False) -> Dict:
        """
        Create multiple Data/MC ratio plots, one for each final state.
        
        Args:
            data_file: Path to data ROOT file
            mc_files: List of MC background ROOT files
            final_state_flags: List of final state selection flags
            mc_labels: Labels for MC backgrounds (auto-generated if None)
            data_scale: Data scaling factor
            mc_scales: MC scaling factors for each file
            base_name: Base name for plots
            output_dir: Output directory
            output_formats: Output formats
            
        Returns:
            Dictionary with results for each final state
        """
        if not final_state_flags:
            return {'success': False, 'error': 'No final state flags provided'}
        
        results = {}
        successful_plots = []
        failed_plots = []
        
        # Determine groupings to create
        groupings = ['ms', 'rs'] if create_both_groupings else [self.grouping_type]
        total_plots = len(final_state_flags) * len(groupings)
        
        print(f"   Creating {total_plots} Data/MC ratio plots ({len(final_state_flags)} final states × {len(groupings)} groupings)...")
        
        # Load all files ONCE at the beginning
        print(f"   Loading data file: {data_file}")
        try:
            data_file_data = self.data_processor.load_file_data(
                data_file, 'data', self.baseline_cuts, data_scale, 1.0
            )
        except Exception as e:
            return {'success': False, 'error': f"Failed to load data file: {str(e)}"}
        
        print(f"   Loading {len(mc_files)} MC files...")
        mc_file_data = []
        mc_file_labels = []
        if mc_scales is None:
            mc_scales = [1.0] * len(mc_files)
        
        for i, mc_file in enumerate(mc_files):
            print(f"     Loading MC file {i+1}/{len(mc_files)}: {mc_file}")
            try:
                mc_scale = mc_scales[i] if i < len(mc_scales) else 1.0
                file_data = self.data_processor.load_file_data(
                    mc_file, 'background', self.baseline_cuts, 1.0, mc_scale
                )
                mc_file_data.append(file_data)
                
                # Generate label
                if mc_labels and i < len(mc_labels):
                    label = mc_labels[i]
                else:
                    import os
                    basename = os.path.basename(mc_file)
                    if "_background" in basename:
                        label = basename.split("_background")[0]
                    else:
                        label = basename.replace(".root", "")
                mc_file_labels.append(label)
                
            except Exception as e:
                print(f"     ✗ Failed to load MC file {mc_file}: {str(e)}")
                self.analysis_summary['errors'].append(f"Failed to load MC file {mc_file}: {str(e)}")
        
        if not mc_file_data:
            return {'success': False, 'error': 'No valid MC files loaded'}
        
        # Now create all plots using cached data
        plot_count = 0
        for final_state in final_state_flags:
            clean_final_state = final_state.replace('pass', '').replace('Selection', '')
            
            for grouping in groupings:
                plot_count += 1
                print(f"   Processing plot {plot_count}/{total_plots}: {final_state} ({grouping.upper()}-grouped)")
                
                # Store original grouping and temporarily change
                original_grouping = self.grouping_type
                self.grouping_type = grouping
                
                # Generate output path for this final state and grouping
                plot_name = f"{base_name}_{clean_final_state}_{grouping}"
                
                try:
                    result = self._create_datamc_ratio_from_cached_data(
                        data_file_data, mc_file_data, mc_file_labels, 
                        final_state, plot_name, normalize
                    )
                    
                    results[f"{final_state}_{grouping}"] = result
                    
                    if result['success']:
                        successful_plots.append(f"{final_state}_{grouping}")
                        print(f"   ✓ Created plot: {plot_name}")
                    else:
                        failed_plots.append({'final_state': f"{final_state}_{grouping}", 'error': result['error']})
                        print(f"   ✗ Failed: {final_state}_{grouping}")
                        
                except Exception as e:
                    error_msg = f"Error creating plot for {final_state}_{grouping}: {str(e)}"
                    failed_plots.append({'final_state': f"{final_state}_{grouping}", 'error': error_msg})
                    self.analysis_summary['errors'].append(error_msg)
                    print(f"   ✗ Exception: {final_state}_{grouping}")
                
                finally:
                    # Restore original grouping
                    self.grouping_type = original_grouping
        
        # Save all canvases to a single ROOT file or individual format files
        if output_path:
            if 'root' in output_formats:
                # Save to single ROOT file
                root_filename = f"{output_path}.root"
                self._save_datamc_canvases_to_root(results, root_filename)
                self.analysis_summary['plots_created'].append(f"ROOT file: {root_filename}")
            
            # Save individual format files if requested (PDF, PNG, etc.)
            non_root_formats = [fmt for fmt in output_formats if fmt != 'root']
            if non_root_formats:
                # Collect all canvases for shared folder saving
                canvases = {}
                for key, result in results.items():
                    if result['success']:
                        final_state_name, grouping = key.rsplit('_', 1)
                        clean_final_state = final_state_name.replace('pass', '').replace('Selection', '')
                        canvas_name = f"{clean_final_state}_{grouping}_datamc"
                        canvases[canvas_name] = result['canvas']
                
                # Save all canvases to shared folder
                self.canvas_maker.save_canvases_to_folder(canvases, output_path, non_root_formats)
        
        return {
            'success': len(failed_plots) == 0,
            'results': results,
            'successful_plots': successful_plots,
            'failed_plots': failed_plots,
            'total_processed': total_plots
        }
    
    def _create_single_datamc_ratio_plot(self, data_file: str, mc_files: List[str], 
                                       final_state: str = None,
                                       mc_labels: List[str] = None,
                                       data_scale: float = 1.0,
                                       mc_scales: List[float] = None,
                                       name: str = "datamc_ratio",
                                       output_path: str = None,
                                       output_formats: List[str] = ['pdf']) -> Dict:
        """
        Create Data/MC ratio plot with stacked backgrounds.
        
        Args:
            data_file: Path to data ROOT file
            mc_files: List of MC background ROOT files
            final_state_flags: Final state selection flags
            mc_labels: Labels for MC backgrounds (auto-generated if None)
            data_scale: Data scaling factor
            mc_scales: MC scaling factors for each file
            name: Base name for plot
            output_path: Output file path
            output_formats: Output formats
            
        Returns:
            Dictionary with canvas and metadata
        """
        if not mc_files:
            return {'success': False, 'error': 'No MC files provided'}
        
        try:
            # Load data file
            data_result = self.create_single_plot(
                data_file, 'data',
                final_state_flags=[final_state] if final_state else None,
                signal_scale=data_scale,
                style_type='data'
            )
            
            if not data_result['success']:
                return {'success': False, 'error': f"Failed to load data: {data_result['error']}"}
            
            data_hist = data_result['histogram']
            
            # Load MC files
            mc_histograms = []
            if mc_scales is None:
                mc_scales = [1.0] * len(mc_files)
            
            for i, mc_file in enumerate(mc_files):
                mc_scale = mc_scales[i] if i < len(mc_scales) else 1.0
                
                mc_result = self.create_single_plot(
                    mc_file, 'background',
                    final_state_flags=[final_state] if final_state else None,
                    background_scale=mc_scale,
                    style_type='background'
                )
                
                if mc_result['success']:
                    mc_hist = mc_result['histogram']
                    
                    # Generate label
                    if mc_labels and i < len(mc_labels):
                        label = mc_labels[i]
                    else:
                        # Extract name from file path
                        import os
                        basename = os.path.basename(mc_file)
                        if "_background" in basename:
                            label = basename.split("_background")[0]
                        else:
                            label = basename.replace(".root", "")
                    
                    mc_histograms.append((mc_hist, label))
                else:
                    self.analysis_summary['errors'].append(f"Failed to load MC file {mc_file}: {mc_result['error']}")
            
            if not mc_histograms:
                return {'success': False, 'error': 'No valid MC histograms created'}
            
            # Create Data/MC ratio canvas
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            canvas = self.canvas_maker.create_datamc_ratio_canvas(
                data_hist, mc_histograms, group_labels, name, normalize, final_state
            )
            
            # Save if output path provided
            if output_path:
                self.canvas_maker.save_canvas(canvas, output_path, output_formats)
                self.analysis_summary['plots_created'].append(f"{output_path} datamc_ratio ({', '.join(output_formats)})")
            
            # Track processing
            self.analysis_summary['files_processed'].append(f"Data/MC ratio: {data_file} vs {len(mc_files)} MC files")
            
            return {
                'success': True,
                'canvas': canvas,
                'data_hist': data_hist,
                'mc_histograms': mc_histograms,
                'group_labels': group_labels,
                'plot_type': 'datamc_ratio'
            }
            
        except Exception as e:
            error_msg = f"Error creating Data/MC ratio plot: {str(e)}"
            self.analysis_summary['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _save_datamc_canvases_to_root(self, results: Dict, output_file: str) -> None:
        """
        Save Data/MC ratio canvases and histograms to a single ROOT file.
        
        Args:
            results: Dictionary of analysis results
            output_file: Output ROOT file path
        """
        root_file = ROOT.TFile(output_file, "RECREATE")
        
        print(f"\nSaving Data/MC ratio plots to: {output_file}")
        
        for key, result in results.items():
            if not result['success']:
                continue
            
            # Extract final state and grouping from key
            final_state_name, grouping = key.rsplit('_', 1)
            clean_final_state = final_state_name.replace('pass', '').replace('Selection', '')
            
            # Create directory for each final state and grouping
            dir_name = f"{clean_final_state}_{grouping}_grouped"
            group_dir = root_file.mkdir(dir_name)
            group_dir.cd()
            
            # Save canvas
            result['canvas'].Write()
            
            # Save histograms if available
            if 'data_hist' in result:
                result['data_hist'].Write()
            if 'mc_histograms' in result:
                for mc_hist, label in result['mc_histograms']:
                    mc_hist.Write()
            
            # Go back to main directory
            root_file.cd()
        
        # Save color palette information for recreation
        color_dir = root_file.mkdir("color_palette")
        color_dir.cd()
        
        # Store the hex colors used
        hex_colors = ["#5A4484", "#347889", "#F4B240", "#E54B26", "#C05780", "#7A68A6", "#2E8B57", "#8B4513"]
        for i, hex_color in enumerate(hex_colors):
            color_index = ROOT.TColor.GetColor(hex_color)
            color_info = ROOT.TNamed(f"color_{i}", f"{hex_color}:{color_index}")
            color_info.Write()
        
        # Save summary information
        summary_dir = root_file.mkdir("summary")
        summary_dir.cd()
        
        # Create summary text
        summary_lines = []
        summary_lines.append(f"Luminosity: {self.luminosity} fb-1")
        summary_lines.append(f"Baseline cuts: {', '.join(self.baseline_cuts)}")
        summary_lines.append(f"Plot type: Data/MC ratio plots")
        summary_lines.append(f"Total plots: {len([r for r in results.values() if r['success']])}")
        summary_lines.append("")
        summary_lines.append("Colors used (hex:index):")
        for i, hex_color in enumerate(hex_colors):
            color_index = ROOT.TColor.GetColor(hex_color)
            summary_lines.append(f"  {hex_color}: {color_index}")
        
        # Save as TNamed object
        summary_content = "\n".join(summary_lines)
        summary_obj = ROOT.TNamed("summary_content", summary_content)
        summary_obj.Write()
        
        # Create a macro to restore colors when opening this file
        restore_macro = '''
{
   // Macro to restore custom colors - run this after opening the ROOT file
   printf("Restoring custom colors...\\n");
   vector<TString> hex_colors = {
      "#5A4484", "#347889", "#F4B240", "#E54B26", 
      "#C05780", "#7A68A6", "#2E8B57", "#8B4513"
   };
   
   for (int i = 0; i < hex_colors.size(); i++) {
      Int_t color_index = TColor::GetColor(hex_colors[i]);
   }
   printf("Custom colors restored!\\n");
   gPad->Update();
}
'''
        macro_obj = ROOT.TNamed("restore_colors_macro", restore_macro)
        macro_obj.Write()
        
        root_file.Close()
        print(f"Successfully saved Data/MC ratio plots to ROOT file: {output_file}")
    
    def _create_datamc_ratio_from_cached_data(self, data_file_data, mc_file_data_list, 
                                            mc_labels, final_state, name, normalize=False):
        """Create Data/MC ratio plot using pre-loaded file data."""
        try:
            # Create data histogram from cached data
            unique_suffix = f"{name}_data"
            data_result = self.create_plot_from_cached_data(
                data_file_data, [final_state] if final_state else None,
                name_suffix=unique_suffix, style_type='data'
            )
            
            if not data_result['success']:
                return {'success': False, 'error': f"Failed to create data histogram: {data_result['error']}"}
            
            data_hist = data_result['histogram']
            
            # Create MC histograms from cached data
            mc_histograms = []
            for i, (mc_file_data, label) in enumerate(zip(mc_file_data_list, mc_labels)):
                unique_mc_suffix = f"{name}_mc_{i}_{label}"
                mc_result = self.create_plot_from_cached_data(
                    mc_file_data, [final_state] if final_state else None,
                    name_suffix=unique_mc_suffix, style_type='background'
                )
                
                if mc_result['success']:
                    mc_hist = mc_result['histogram']
                    mc_histograms.append((mc_hist, label))
                else:
                    self.analysis_summary['errors'].append(f"Failed to create MC histogram for {label}: {mc_result['error']}")
            
            if not mc_histograms:
                return {'success': False, 'error': 'No valid MC histograms created from cached data'}
            
            # Create Data/MC ratio canvas
            group_labels = self.data_processor.get_group_labels(self.grouping_type)
            canvas = self.canvas_maker.create_datamc_ratio_canvas(
                data_hist, mc_histograms, group_labels, name, normalize, final_state
            )
            
            return {
                'success': True,
                'canvas': canvas,
                'data_hist': data_hist,
                'mc_histograms': mc_histograms,
                'group_labels': group_labels,
                'plot_type': 'datamc_ratio',
                'final_state': final_state
            }
            
        except Exception as e:
            error_msg = f"Error creating Data/MC ratio plot from cached data: {str(e)}"
            return {'success': False, 'error': error_msg}
    
    def recreate_original_plot(self, file_path: str, file_type: str,
                              final_state_selection: str = None,
                              signal_scale: float = 1.0, background_scale: float = 1.0,
                              output_file: str = None) -> Dict:
        """
        Recreate the exact plot from the original unrolled_yield_analysis.py.
        Saves both Ms and Rs grouped plots to a single ROOT file.
        
        Args:
            file_path: Path to ROOT file
            file_type: 'signal', 'background', or 'data'
            final_state_selection: Single final state selection flag
            signal_scale: Signal scaling factor
            background_scale: Background scaling factor
            output_file: Output ROOT file path (default: auto-generated)
            
        Returns:
            Dictionary with both Ms and Rs grouped plots
        """
        results = {}
        canvases = {}
        
        # Load file data once
        try:
            final_state_flags = [final_state_selection] if final_state_selection else None
            file_data = self.data_processor.load_file_data(
                file_path, file_type, self.baseline_cuts, signal_scale, background_scale
            )
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to load file data: {str(e)}",
                'results': {},
                'canvases': {}
            }
        
        # Create both Ms-grouped and Rs-grouped plots from cached data
        for grouping in ['ms', 'rs']:
            # Temporarily change grouping
            original_grouping = self.grouping_type
            self.grouping_type = grouping
            
            # Create plot from cached data
            result = self.create_plot_from_cached_data(
                file_data,
                final_state_flags=final_state_flags,
                show_error_band=True
            )
            
            results[f'{grouping}_grouped'] = result
            if result['success']:
                canvases[f'{grouping}_grouped'] = result['canvas']
            
            # Restore original grouping
            self.grouping_type = original_grouping
        
        # Save to single ROOT file
        if output_file is None:
            file_basename = file_path.split('/')[-1].split('.')[0]
            final_state_suffix = f"_{final_state_selection}" if final_state_selection else ""
            output_file = f"{file_basename}_unrolled_plots{final_state_suffix}.root"
        
        if canvases:
            self.save_to_root_file(canvases, results, output_file)
            self.analysis_summary['plots_created'].append(f"ROOT file: {output_file}")
        
        return {
            'success': all(result.get('success', False) for result in results.values()),
            'results': results,
            'output_file': output_file,
            'canvases': canvases
        }
    
    def batch_process_files(self, file_list: List[Dict], 
                           final_state_flags: List[str] = None,
                           output_dir: str = '.', 
                           output_formats: List[str] = ['pdf']) -> Dict:
        """
        Process multiple files in batch.
        
        Args:
            file_list: List of dicts with 'path', 'type', 'scale' keys
            final_state_flags: Final state selection flags
            output_dir: Output directory
            output_formats: Output formats
            
        Returns:
            Summary of batch processing
        """
        successful_plots = []
        failed_plots = []
        
        for file_config in file_list:
            file_basename = file_config['path'].split('/')[-1].split('.')[0]
            output_path = f"{output_dir}/{file_basename}_{self.grouping_type}"
            
            result = self.create_single_plot(
                file_config['path'], file_config['type'],
                final_state_flags=final_state_flags,
                signal_scale=file_config.get('scale', 1.0),
                output_path=output_path,
                output_formats=output_formats
            )
            
            if result['success']:
                successful_plots.append(file_config['path'])
            else:
                failed_plots.append({'path': file_config['path'], 'error': result['error']})
        
        return {
            'success': len(failed_plots) == 0,
            'successful_plots': successful_plots,
            'failed_plots': failed_plots,
            'total_processed': len(file_list)
        }
    
    def save_to_root_file(self, canvases: Dict, results: Dict, output_file: str) -> None:
        """
        Save canvases and histograms to a single ROOT file.
        
        Args:
            canvases: Dictionary of canvases to save
            results: Dictionary of analysis results
            output_file: Output ROOT file path
        """
        root_file = ROOT.TFile(output_file, "RECREATE")
        
        print(f"\nSaving plots to: {output_file}")
        
        for grouping_key, canvas in canvases.items():
            # Create directory for each grouping
            grouping_name = grouping_key.replace('_grouped', '')
            dir_name = f"{grouping_name}_grouped"
            group_dir = root_file.mkdir(dir_name)
            group_dir.cd()
            
            # Save canvas (skip if None)
            if canvas is not None:
                canvas.Write()
            else:
                print(f"WARNING: Skipping {dir_name} - canvas is None")
            
            # Save histogram if available
            if grouping_key in results and results[grouping_key]['success']:
                result = results[grouping_key]
                if 'histogram' in result:
                    result['histogram'].Write()
                if 'error_band' in result and result['error_band'] is not None:
                    result['error_band'].Write()
            
            # Go back to main directory
            root_file.cd()
        
        # Save summary information
        summary_dir = root_file.mkdir("summary")
        summary_dir.cd()
        
        # Create summary text
        summary_text = ROOT.TNamed("analysis_summary", "Analysis Summary")
        summary_lines = []
        summary_lines.append(f"Luminosity: {self.luminosity} fb-1")
        summary_lines.append(f"Baseline cuts: {', '.join(self.baseline_cuts)}")
        
        for grouping_key, result in results.items():
            if result['success']:
                # Handle different result structures
                if 'total_yield' in result:
                    summary_lines.append(f"{grouping_key} total yield: {result['total_yield']:.2f}")
                    summary_lines.append(f"{grouping_key} bins with data: {result['n_bins_with_data']}/9")
                elif 'histograms' in result:
                    # For multi-final-state plots, calculate total from histograms
                    total_yield = sum(hist.Integral() for hist in result['histograms'])
                    summary_lines.append(f"{grouping_key} total yield: {total_yield:.2f}")
                    summary_lines.append(f"{grouping_key} histograms: {len(result['histograms'])}")
        
        # Save as TNamed object
        summary_content = "\n".join(summary_lines)
        summary_obj = ROOT.TNamed("summary_content", summary_content)
        summary_obj.Write()
        
        root_file.Close()
        print(f"Successfully saved ROOT file: {output_file}")
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "="*80)
        print("UNROLLED PLOTTER SUMMARY")
        print("="*80)
        
        print(f"\n🎨 CONFIGURATION:")
        print("-" * 50)
        print(f"    • Luminosity: {self.luminosity:.1f} fb⁻¹")
        print(f"    • Grouping type: {self.grouping_type}")
        print(f"    • Baseline cuts: {', '.join(self.baseline_cuts)}")
        
        if self.analysis_summary['files_processed']:
            print(f"\n📁 FILES PROCESSED ({len(self.analysis_summary['files_processed'])}):")
            print("-" * 50)
            for file_info in self.analysis_summary['files_processed']:
                print(f"    • {file_info}")
        
        if self.analysis_summary['plots_created']:
            print(f"\n📊 PLOTS CREATED ({len(self.analysis_summary['plots_created'])}):")
            print("-" * 50)
            for plot in self.analysis_summary['plots_created']:
                print(f"    • {plot}")
        
        if self.analysis_summary['errors']:
            print(f"\n⚠️  ERRORS ({len(self.analysis_summary['errors'])}):")
            print("-" * 50)
            for error in self.analysis_summary['errors']:
                print(f"    • {error}")
        
        # Print component summaries
        self.data_processor.print_processing_summary()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
    

def main():
    """Extended CLI for all types of unrolled plots."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced unrolled yield plotting with multiple plot types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file, single final state (original)
  python unrolled_plotter.py --signal signal.root --final-state passNHad1SelectionSRLoose
  
  # Single file, multiple final states  
  python unrolled_plotter.py --signal signal.root --final-states passNHad1SelectionSRLoose passNHad1SelectionSRTight
  
  # Multiple files comparison
  python unrolled_plotter.py --signal signal1.root signal2.root --background bkg.root --final-state passNHad1SelectionSRLoose
  
  # Data/MC ratio plot (single final state)
  python unrolled_plotter.py --data data.root --background bkg1.root bkg2.root --final-state passNHad1SelectionSRLoose --datamc-ratio
  
  # Data/MC ratio plots (multiple final states)
  python unrolled_plotter.py --data data.root --background bkg1.root bkg2.root --final-states passNHad1SelectionSRLoose passNLep1SelectionSRLoose --datamc-ratio
        """)
    
    # Input files (like original script)
    parser.add_argument('--signal', type=str, nargs='*', default=[],
                       help='Signal ROOT files')
    parser.add_argument('--background', type=str, nargs='*', default=[],
                       help='Background ROOT files') 
    parser.add_argument('--data', type=str, nargs='*', default=[],
                       help='Data ROOT files')
    
    # Data/MC ratio plot option
    parser.add_argument('--datamc-ratio', action='store_true',
                       help='Create Data/MC ratio plot with stacked backgrounds')
    
    # Final state selections
    parser.add_argument('--final-state', type=str, 
                       help='Single final state selection flag')
    parser.add_argument('--final-states', type=str, nargs='+',
                       help='Multiple final state flags for comparison (single file only)')
    
    # Plot configuration
    parser.add_argument('--grouping', type=str, default='ms', choices=['ms', 'rs'],
                       help='Grouping type: ms or rs (default: ms)')
    parser.add_argument('--luminosity', type=float, default=400.0,
                       help='Luminosity in fb-1 (default: 400)')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Custom legend labels')
    
    # Output options
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--output-formats', type=str, nargs='+', default=['root'],
                       choices=['pdf', 'png', 'root', 'eps', 'svg'],
                       help='Output formats (default: root)')
    parser.add_argument('--name', type=str, default='unrolled_plot',
                       help='Base name for plot (default: unrolled_plot)')
    
    # Scaling factors
    parser.add_argument('--signal-scale', type=float, default=1.0,
                       help='Signal scaling factor (default: 1.0)')
    parser.add_argument('--background-scale', type=float, default=1.0,
                       help='Background scaling factor (default: 1.0)')
    
    # Advanced options
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize histograms for shape comparison')
    parser.add_argument('--no-error-bands', action='store_true',
                       help='Disable error bands')
    parser.add_argument('--use-markers', action='store_true',
                       help='Use offset markers instead of overlapping lines for multiple curves')
    
    args = parser.parse_args()
    
    # Validate inputs
    total_files = len(args.signal) + len(args.background) + len(args.data)
    if total_files == 0:
        parser.error("Must provide at least one input file")
    
    if args.final_states and total_files > 1 and not args.datamc_ratio:
        parser.error("--final-states can only be used with a single input file (unless using --datamc-ratio)")
    
    # Create plotter
    plotter = UnrolledPlotter(
        luminosity=args.luminosity,
        grouping_type=args.grouping
    )
    
    print("🚀 Starting unrolled yield plotting...")
    print(f"   📊 Grouping: {args.grouping.upper()}")
    print(f"   💡 Luminosity: {args.luminosity} fb⁻¹")
    
    # Determine plot mode and execute
    if args.datamc_ratio:
        # Data/MC ratio plot mode
        print("   📊 Mode: Data/MC ratio plot with stacked backgrounds")
        
        # Validate inputs for Data/MC ratio
        if not args.data or not args.background:
            parser.error("--datamc-ratio requires both --data and --background files")
        if len(args.data) != 1:
            parser.error("--datamc-ratio requires exactly one data file")
        if len(args.background) == 0:
            parser.error("--datamc-ratio requires at least one background file")
        
        # Handle multiple final states for Data/MC ratio
        if args.final_states:
            print(f"   Creating Data/MC ratio plots for {len(args.final_states)} final states")
            
            result = plotter.create_multi_datamc_ratio_plots(
                data_file=args.data[0],
                mc_files=args.background,
                final_state_flags=args.final_states,
                mc_labels=args.labels,
                data_scale=1.0,  # Data is not scaled
                mc_scales=[args.background_scale] * len(args.background),
                base_name=args.name,
                output_path=args.output_file,
                output_formats=args.output_formats,
                normalize=args.normalize
            )
        else:
            # Single final state (or no final state)
            result = plotter.create_datamc_ratio_plot(
                data_file=args.data[0],
                mc_files=args.background,
                final_state_flags=[args.final_state] if args.final_state else None,
                mc_labels=args.labels,
                data_scale=1.0,  # Data is not scaled
                mc_scales=[args.background_scale] * len(args.background),
                name=args.name,
                output_path=args.output_file,
                output_formats=args.output_formats,
                normalize=args.normalize
            )
        
    elif args.final_states:
        # Multiple final states from single file - create both Ms and Rs grouped versions
        if total_files != 1:
            parser.error("--final-states requires exactly one input file")
        
        # Get the single file and its type
        if args.signal:
            file_path, file_type = args.signal[0], 'signal'
            scale = args.signal_scale
        elif args.background:
            file_path, file_type = args.background[0], 'background'
            scale = args.background_scale
        else:
            file_path, file_type = args.data[0], 'data'
            scale = 1.0
        
        print(f"   🎯 Mode: Multiple final states ({len(args.final_states)} states)")
        print("   📊 Creating both Ms-grouped and Rs-grouped plots")
        
        # Load file data once for efficiency
        file_data = plotter.data_processor.load_file_data(
            file_path, file_type, plotter.baseline_cuts, 
            signal_scale=scale if file_type == 'signal' else 1.0,
            background_scale=scale if file_type == 'background' else 1.0
        )
        
        # Create both Ms and Rs grouped plots
        results = {}
        canvases = {}
        
        for grouping in ['ms', 'rs']:
            # Temporarily change grouping
            original_grouping = plotter.grouping_type
            plotter.grouping_type = grouping
            
            print(f"   Creating {grouping.upper()}-grouped comparison plot...")
            
            # Create histograms for each final state using cached data
            histograms = []
            plot_labels = []
            original_yields = []  # Store original yields before normalization
            
            for i, final_state in enumerate(args.final_states):
                # Calculate yields for this final state
                yields_2d, errors_2d = plotter.data_processor.calculate_2d_yields_from_data(
                    file_data, [final_state]
                )
                
                # Unroll to 1D
                yields_1d, errors_1d, bin_labels = plotter.data_processor.unroll_2d_to_1d(
                    yields_2d, errors_2d, grouping
                )
                
                # Create histogram
                hist_name = f"{args.name}_{grouping}_{final_state}_{i}"
                hist = plotter.histogram_maker.create_histogram(
                    yields_1d, errors_1d, bin_labels, hist_name, "", grouping
                )
                
                # Apply comparison styling with color index
                plotter.histogram_maker.style_histogram(hist, 'comparison', color_index=i)
                
                # Store original yield BEFORE normalization
                original_yields.append(hist.Integral())
                
                # Normalize if requested
                if args.normalize:
                    hist = plotter.histogram_maker.normalize_histogram(hist, 'unity')
                
                histograms.append(hist)
                
                # Generate label
                if args.labels and i < len(args.labels):
                    plot_labels.append(args.labels[i])
                else:
                    # Convert to SV notation: {N}SV_{flavor}^{selection}
                    clean_label = plotter.canvas_maker._format_sv_label(final_state)
                    plot_labels.append(clean_label)
            
            # Get group labels for both cases
            group_labels = plotter.data_processor.get_group_labels(grouping)
            
            # Create comparison canvas
            canvas_name = f"{args.name}_{grouping}_comparison"
            
            if args.use_markers:
                canvas = plotter.canvas_maker.create_comparison_canvas_with_markers(
                    histograms, plot_labels, group_labels, canvas_name, marker_styles=None, original_yields=original_yields
                )
            else:
                canvas = plotter.canvas_maker.create_comparison_canvas(
                    histograms, plot_labels, group_labels, canvas_name
                )
            
            results[f'{grouping}_grouped'] = {
                'success': True,
                'canvas': canvas,
                'histograms': histograms,
                'labels': plot_labels,
                'group_labels': group_labels
            }
            canvases[f'{grouping}_grouped'] = canvas
            
            # Restore original grouping
            plotter.grouping_type = original_grouping
        
        # Save both plots to a single ROOT file or separate files based on format
        if args.output_file:
            if 'root' in args.output_formats:
                # Save both to single ROOT file
                plotter.save_to_root_file(canvases, results, f"{args.output_file}.root")
                plotter.analysis_summary['plots_created'].append(f"ROOT file: {args.output_file}.root")
            
            # Save separate files for other formats
            non_root_formats = [fmt for fmt in args.output_formats if fmt != 'root']
            if non_root_formats:
                # Collect all canvases for shared folder saving
                comparison_canvases = {
                    f"{args.name}_{grouping}_comparison": canvases[f'{grouping}_grouped'] 
                    for grouping in ['ms', 'rs']
                }
                
                # Save all canvases to shared folder
                plotter.canvas_maker.save_canvases_to_folder(comparison_canvases, args.output_file, non_root_formats)
        else:
            # Auto-generate output file name
            file_basename = file_path.split('/')[-1].split('.')[0]
            output_file = f"{file_basename}_multi_final_states.root"
            if 'root' in args.output_formats:
                plotter.save_to_root_file(canvases, results, output_file)
                plotter.analysis_summary['plots_created'].append(f"ROOT file: {output_file}")
        
        # Track processing
        plotter.analysis_summary['files_processed'].append(f"Multi-final-states from {file_type}: {file_path}")
        
        # Create combined result
        result = {
            'success': all(result.get('success', False) for result in results.values()),
            'results': results,
            'canvases': canvases,
            'both_groupings_created': True
        }
        
    elif total_files > 1 or (args.signal and len(args.signal) > 1):
        # Multiple files comparison
        print(f"   📁 Mode: Multiple files comparison ({total_files} files)")
        
        file_configs = []
        
        # Add signal files
        for file_path in args.signal:
            file_configs.append({
                'path': file_path,
                'type': 'signal',
                'scale': args.signal_scale
            })
        
        # Add background files  
        for file_path in args.background:
            file_configs.append({
                'path': file_path,
                'type': 'background', 
                'scale': args.background_scale
            })
            
        # Add data files
        for file_path in args.data:
            file_configs.append({
                'path': file_path,
                'type': 'data',
                'scale': 1.0
            })
        
        if args.use_markers:
            result = plotter.create_comparison_plot_with_markers(
                file_configs=file_configs,
                final_state_flags=[args.final_state] if args.final_state else None,
                labels=args.labels,
                name=args.name,
                output_path=args.output_file,
                output_formats=args.output_formats,
                normalize=args.normalize
            )
        else:
            result = plotter.create_comparison_plot(
                file_configs=file_configs,
                final_state_flags=[args.final_state] if args.final_state else None,
                labels=args.labels,
                name=args.name,
                output_path=args.output_file,
                output_formats=args.output_formats,
                normalize=args.normalize
            )
        
    else:
        # Single file (original functionality)
        print("   📄 Mode: Single plot (original functionality)")
        
        # Get the single file and its type
        if args.signal:
            file_path, file_type = args.signal[0], 'signal'
            scale = args.signal_scale
        elif args.background:
            file_path, file_type = args.background[0], 'background'
            scale = args.background_scale
        else:
            file_path, file_type = args.data[0], 'data'
            scale = 1.0
        
        result = plotter.recreate_original_plot(
            file_path=file_path,
            file_type=file_type,
            final_state_selection=args.final_state,
            signal_scale=scale if file_type == 'signal' else 1.0,
            background_scale=scale if file_type == 'background' else 1.0,
            output_file=args.output_file
        )
    
    # Report results
    if result['success']:
        print("\n✅ SUCCESS!")
        if 'output_file' in result:
            print(f"📁 Saved to: {result['output_file']}")
        if 'canvas' in result:
            print("📊 Plot created with all decorations")
        if 'histograms' in result:
            print(f"📈 {len(result['histograms'])} histograms overlaid")
        print(f"🎨 Formats: {', '.join(args.output_formats)}")
    else:
        print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
        return 1
    
    # Print summary
    plotter.print_summary()
    return 0

if __name__ == "__main__":
    main()