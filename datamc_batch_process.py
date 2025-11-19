#!/usr/bin/env python3
"""
Multi-Region SV Variables Analysis

Enhanced multi-region analysis that creates SV variable Data/MC comparison plots
for each region/multiplicity combination. Supports both ROOT file and PDF folder output.

Integrates with existing multi_region_analysis_fast.py framework while adding
comprehensive SV variable plotting capabilities.
"""

import ROOT
import argparse
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Import our modular utilities
from sv_plotting_utils import (
    SVVariablesPlotter, RegionSelector, DataMCComparisonPlotter,
    create_data_mc_comparison_plots, save_plots_to_root_directory, save_plots_to_pdf_folder
)

# Enable batch mode and disable title/stat boxes
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0)

class FileCache:
    """Cache for ROOT file data to avoid repeated reads."""
    def __init__(self):
        self.cache = {}
    
    def get_file_data(self, file_path: str) -> Dict:
        """Get cached file data or load it if not cached."""
        if file_path not in self.cache:
            plotter = SVVariablesPlotter()
            self.cache[file_path] = plotter.load_branches_for_sv_plotting(file_path)
        return self.cache[file_path]

class MultiRegionSVAnalyzer:
    """Enhanced multi-region analyzer with SV variable plotting."""
    
    def __init__(self, luminosity: float = 400.0, parallel_processing: bool = True, 
                 n_processes: Optional[int] = None):
        """
        Initialize the multi-region SV analyzer.
        
        Args:
            luminosity: Integrated luminosity in fb-1
            parallel_processing: Whether to use parallel processing
            n_processes: Number of processes (None for auto-detect)
        """
        self.luminosity = luminosity
        self.parallel_processing = parallel_processing
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.file_cache = FileCache()
        
        # Define SV type configurations (adapted from multi_region_analysis_fast.py)
        self.sv_configs = {
            'HadronSV': {
                'regions': [
                    {'name': 'dxySig_lt90', 'dxySig_cut': '<:90', 'label': 'S_{xy} < 90', 'is_signal_region': False},
                    {'name': 'dxySig_90to300', 'dxySig_cut': '>=:90,<:300', 'label': '90 < S_{xy} < 300', 'is_signal_region': False},
                    {'name': 'dxySig_300to1000', 'dxySig_cut': '>=:300,<:1000', 'label': '300 < S_{xy} < 1000', 'is_signal_region': False},
                    {'name': 'dxySig_lt1000', 'dxySig_cut': '<:1000', 'label': 'S_{xy} < 1000 (integrated sideband)', 'is_signal_region': False},
                    {'name': 'dxySig_gt1000', 'dxySig_cut': '>:1000', 'label': 'S_{xy} > 1000', 'is_signal_region': True}
                ],
                'multiplicities': [
                    {'name': 'nHad1', 'cuts': ['SV_nHadronic:==:1'], 'label': 'nHadronic = 1'},
                    {'name': 'nHadGT1', 'cuts': ['SV_nHadronic:>:1'], 'label': 'nHadronic > 1'}
                ]
            },
            'LeptonSV': {
                'regions': [
                    {'name': 'dxySig_lt40', 'dxySig_cut': '<:40', 'label': 'S_{xy} < 40', 'is_signal_region': False},
                    {'name': 'dxySig_40to70', 'dxySig_cut': '>=:40,<:70', 'label': '40 < S_{xy} < 70', 'is_signal_region': False},
                    {'name': 'dxySig_70to1000', 'dxySig_cut': '>=:70,<:1000', 'label': '70 < S_{xy} < 1000', 'is_signal_region': False},
                    {'name': 'dxySig_lt1000', 'dxySig_cut': '<:1000', 'label': 'S_{xy} < 1000 (integrated sideband)', 'is_signal_region': False},
                    {'name': 'dxySig_gt1000', 'dxySig_cut': '>:1000', 'label': 'S_{xy} > 1000', 'is_signal_region': True}
                ],
                'multiplicities': [
                    {'name': 'nLep1', 'cuts': ['SV_nLeptonic:==:1'], 'label': 'nLeptonic = 1'},
                    {'name': 'nLepGT1', 'cuts': ['SV_nLeptonic:>:1'], 'label': 'nLeptonic > 1'}
                ]
            },
            'AnySV': {
                'regions': [
                    {'name': 'any_dxySig_lt1000', 'dxySig_cut': '<:1000', 'label': 'Any S_{xy} < 1000', 'is_signal_region': False},
                    {'name': 'any_dxySig_gt1000', 'dxySig_cut': '>:1000', 'label': 'Any S_{xy} > 1000', 'is_signal_region': True}
                ],
                'multiplicities': [
                    {'name': 'nLepAndHadGE1', 'cuts': ['SV_nLeptonic:>=:1', 'SV_nHadronic:>=:1'], 'label': 'nLeptonic #geq 1 & nHadronic #geq 1'}
                ]
            },
            '0SV': {
                'regions': [
                    {'name': 'inclusive', 'label': 'Inclusive', 'is_signal_region': False}
                ],
                'multiplicities': [
                    {'name': 'nSV0', 'cuts': ['SV_nLeptonic:==:0', 'SV_nHadronic:==:0'], 'label': 'nLeptonic = 0 & nHadronic = 0'}
                ]
            }
        }
        
        # Analysis summary
        self.analysis_summary = {
            'total_combinations': 0,
            'successful_combinations': 0,
            'failed_combinations': 0,
            'blinded_combinations': 0,
            'files_processed': 0,
            'plots_created': 0
        }
    
    def should_blind_data(self, region: Dict) -> bool:
        """Check if data should be blinded for this region."""
        return region.get('is_signal_region', False)
    
    def process_single_combination(self, sv_type: str, multiplicity: Dict, region: Dict,
                                 data_files_data: List[Dict], mc_files_data: List[Dict], 
                                 mc_file_names: List[str], mc_file_scaling: List[float],
                                 variables_to_plot: Optional[List[str]] = None,
                                 extra_cuts: Optional[str] = None, normalize: bool = False) -> Dict:
        """Process a single SV type/multiplicity/region combination."""
        
        try:
            # Check if data should be blinded
            blind_data = self.should_blind_data(region)
            
            # Skip plot creation for blinded signal regions
            if blind_data:
                return {
                    'success': True,
                    'canvases': {},
                    'blinded': True,
                    'n_plots': 0
                }
            
            # Create SV variable plots for this combination
            canvases = create_data_mc_comparison_plots(
                data_files_data, mc_files_data, mc_file_names, mc_file_scaling,
                region, multiplicity, sv_type, variables_to_plot, blind_data, self.luminosity, extra_cuts, normalize
            )
            
            if canvases:
                self.analysis_summary['plots_created'] += len(canvases)
                return {
                    'success': True,
                    'canvases': canvases,
                    'blinded': False,
                    'n_plots': len(canvases)
                }
            else:
                return {
                    'success': False,
                    'error': 'No valid plots created'
                }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_all_combinations(self) -> List[Tuple]:
        """Generate all possible SV type/multiplicity/region combinations."""
        combinations = []
        for sv_type in self.sv_configs:
            for multiplicity in self.sv_configs[sv_type]['multiplicities']:
                for region in self.sv_configs[sv_type]['regions']:
                    combinations.append((sv_type, multiplicity, region))
        return combinations
    
    def save_to_root_file(self, all_results: Dict, output_file: str) -> None:
        """Save all results to organized ROOT file structure."""
        root_file = ROOT.TFile(output_file, "RECREATE")
        
        print(f"\nSaving results to ROOT file: {output_file}")
        
        for sv_type in all_results:
            sv_dir = root_file.mkdir(sv_type)
            sv_dir.cd()
            
            for multiplicity_name in all_results[sv_type]:
                mult_dir = sv_dir.mkdir(multiplicity_name)
                mult_dir.cd()
                
                for region_name in all_results[sv_type][multiplicity_name]:
                    region_dir = mult_dir.mkdir(region_name)
                    region_dir.cd()
                    
                    # Create sv_plots subdirectory
                    sv_plots_dir = region_dir.mkdir("sv_plots")
                    
                    result = all_results[sv_type][multiplicity_name][region_name]
                    print(f"  Saving {sv_type}/{multiplicity_name}/{region_name}")
                    if result['success']:
                        save_plots_to_root_directory(result['canvases'], sv_plots_dir)
                    elif result.get('blinded', False):
                        # Add placeholder for blinded regions
                        sv_plots_dir.cd()
                        placeholder = ROOT.TNamed("blinded_sv_plots", "SV plots blinded in signal region")
                        placeholder.Write()
                    print(f"    ✓ Saved {len(result.get('canvases', {}))} plots")
        
        root_file.Close()
        print(f"Successfully saved ROOT file: {output_file}")
        
        # Clear ROOT's internal canvas list to prevent TList::Clear errors
        try:
            # Disable ROOT's automatic cleanup to prevent double deletion errors
            ROOT.gROOT.SetMustClean(False)
            
            # Clear the list of canvases without deleting objects
            canvas_list = ROOT.gROOT.GetListOfCanvases()
            if canvas_list:
                canvas_list.Clear("nodelete")
            
            # Python garbage collection
            import gc
            gc.collect()
        except:
            pass
    
    def save_to_pdf_folders(self, all_results: Dict, base_pdf_dir: str) -> None:
        """Save all results to PDF folder structure."""
        print(f"\nSaving results to PDF folders: {base_pdf_dir}")
        
        for sv_type in all_results:
            for multiplicity_name in all_results[sv_type]:
                for region_name in all_results[sv_type][multiplicity_name]:
                    # Create folder path: base_dir/SvType/multiplicity/region/sv_plots/
                    folder_path = os.path.join(base_pdf_dir, sv_type, multiplicity_name, region_name, "sv_plots")
                    
                    result = all_results[sv_type][multiplicity_name][region_name]
                    if result['success']:
                        save_plots_to_pdf_folder(result['canvases'], folder_path)
                        print(f"  Saved {result['n_plots']} plots to {folder_path}")
                    elif result.get('blinded', False):
                        # Create folder but add a note about blinding
                        os.makedirs(folder_path, exist_ok=True)
                        with open(os.path.join(folder_path, "BLINDED_SIGNAL_REGION.txt"), 'w') as f:
                            f.write("SV variable plots are blinded in signal regions (dxySig > 1000)")
        
        print(f"Successfully saved PDF folders to: {base_pdf_dir}")
    
    def run_analysis(self, data_files: List[str], mc_files: List[str], 
                    mc_scaling_factors: Optional[List[float]] = None,
                    variables_to_plot: Optional[List[str]] = None,
                    output_root_file: Optional[str] = None,
                    pdf_output: bool = False, pdf_base_dir: str = "sv_plots_output",
                    extra_cuts: Optional[str] = None, normalize: bool = False) -> Dict:
        """
        Run the complete multi-region SV analysis.
        
        Args:
            data_files: List of data ROOT file paths
            mc_files: List of MC ROOT file paths
            mc_scaling_factors: Optional list of scaling factors for each MC file (default: 1.0 for all)
            variables_to_plot: Optional list of specific variables to plot
            output_root_file: ROOT file output path (if desired)
            pdf_output: Whether to save PDF folder structure
            pdf_base_dir: Base directory for PDF output
            extra_cuts: Optional string with additional cuts (e.g., "selCMet > 200 && rjr_Ms < 5000")
            normalize: Whether to normalize data and MC to their total integrals for shape comparison
            
        Returns:
            Dictionary of all results organized by sv_type/multiplicity/region
        """
        
        # Set default scaling factors if not provided
        if mc_scaling_factors is None:
            mc_scaling_factors = [1.0] * len(mc_files)
        elif len(mc_scaling_factors) != len(mc_files):
            raise ValueError(f"Number of scaling factors ({len(mc_scaling_factors)}) must match number of MC files ({len(mc_files)})")
        
        print(f"Starting Multi-Region SV Analysis...")
        print(f"Data files: {len(data_files)}")
        print(f"MC files: {len(mc_files)}")
        print(f"Luminosity: {self.luminosity} fb^-1")
        print(f"Variables to plot: {variables_to_plot or 'All'}")
        if extra_cuts:
            print(f"Extra cuts: {extra_cuts}")
        
        # Show MC scaling info
        scaled_files = [f for i, f in enumerate(mc_files) if mc_scaling_factors[i] != 1.0]
        if scaled_files:
            print(f"MC files with custom scaling: {len(scaled_files)}")
            for i, (mc_file, scale) in enumerate(zip(mc_files, mc_scaling_factors)):
                if scale != 1.0:
                    print(f"  {mc_file}: {scale}x")
        
        # Load all file data using cache
        print("\nLoading data files...")
        data_files_data = []
        for data_file in data_files:
            if os.path.exists(data_file):
                file_data = self.file_cache.get_file_data(data_file)
                if file_data:
                    data_files_data.append(file_data)
                    print(f"  Loaded: {data_file}")
            else:
                print(f"  Warning: File not found: {data_file}")
        
        print("\nLoading MC files...")
        mc_files_data = []
        mc_file_names = []
        mc_file_scaling = []
        for i, mc_file in enumerate(mc_files):
            if os.path.exists(mc_file):
                file_data = self.file_cache.get_file_data(mc_file)
                if file_data:
                    mc_files_data.append(file_data)
                    mc_file_names.append(mc_file)
                    mc_file_scaling.append(mc_scaling_factors[i])
                    scale_info = f" (scale: {mc_scaling_factors[i]}x)" if mc_scaling_factors[i] != 1.0 else ""
                    print(f"  Loaded: {mc_file}{scale_info}")
            else:
                print(f"  Warning: File not found: {mc_file}")
        
        self.analysis_summary['files_processed'] = len(data_files_data) + len(mc_files_data)
        
        # Generate all combinations
        all_combinations = self.generate_all_combinations()
        self.analysis_summary['total_combinations'] = len(all_combinations)
        
        print(f"\nProcessing {len(all_combinations)} region combinations...")
        
        # Initialize results structure
        all_results = {}
        for sv_type in self.sv_configs:
            all_results[sv_type] = {}
            for multiplicity in self.sv_configs[sv_type]['multiplicities']:
                all_results[sv_type][multiplicity['name']] = {}
                for region in self.sv_configs[sv_type]['regions']:
                    all_results[sv_type][multiplicity['name']][region['name']] = {}
        
        # Process combinations
        if self.parallel_processing and not pdf_output:  # Disable parallel for PDF to avoid ROOT graphics issues
            print(f"Using parallel processing with {self.n_processes} workers...")
            self._process_combinations_parallel(all_combinations, data_files_data, mc_files_data, 
                                              mc_file_names, mc_file_scaling, variables_to_plot, all_results, extra_cuts, normalize)
        else:
            print("Using sequential processing...")
            self._process_combinations_sequential(all_combinations, data_files_data, mc_files_data,
                                                mc_file_names, mc_file_scaling, variables_to_plot, all_results, extra_cuts, normalize)
        
        # Save results
        if output_root_file:
            self.save_to_root_file(all_results, output_root_file)
        
        if pdf_output:
            self.save_to_pdf_folders(all_results, pdf_base_dir)
        
        # Print summary
        self.print_analysis_summary()
        
        return all_results
    
    def _process_combinations_sequential(self, combinations: List[Tuple], data_files_data: List[Dict],
                                       mc_files_data: List[Dict], mc_file_names: List[str],
                                       mc_file_scaling: List[float], variables_to_plot: Optional[List[str]], 
                                       all_results: Dict, extra_cuts: Optional[str] = None, normalize: bool = False) -> None:
        """Process combinations sequentially."""
        
        with tqdm(total=len(combinations), desc="Processing combinations", unit="combination") as pbar:
            for sv_type, multiplicity, region in combinations:
                pbar.set_description(f"Processing {sv_type}/{multiplicity['name']}/{region['name']}")
                
                result = self.process_single_combination(
                    sv_type, multiplicity, region, data_files_data, 
                    mc_files_data, mc_file_names, mc_file_scaling, variables_to_plot, extra_cuts, normalize
                )
                
                # Store result
                all_results[sv_type][multiplicity['name']][region['name']] = result
                
                # Update summary
                if result['success'] and result.get('blinded', False):
                    self.analysis_summary['blinded_combinations'] += 1
                elif result['success']:
                    self.analysis_summary['successful_combinations'] += 1
                else:
                    self.analysis_summary['failed_combinations'] += 1
                
                pbar.update(1)
    
    def _process_combinations_parallel(self, combinations: List[Tuple], data_files_data: List[Dict],
                                     mc_files_data: List[Dict], mc_file_names: List[str],
                                     mc_file_scaling: List[float], variables_to_plot: Optional[List[str]], 
                                     all_results: Dict, extra_cuts: Optional[str] = None, normalize: bool = False) -> None:
        """Process combinations in parallel."""
        
        from functools import partial
        from concurrent.futures import as_completed
        
        # Create partial function
        process_func = partial(
            self.process_single_combination,
            data_files_data=data_files_data,
            mc_files_data=mc_files_data,
            mc_file_names=mc_file_names,
            mc_file_scaling=mc_file_scaling,
            variables_to_plot=variables_to_plot,
            extra_cuts=extra_cuts,
            normalize=normalize
        )
        
        with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
            # Submit all tasks
            future_to_combination = {
                executor.submit(process_func, sv_type, multiplicity, region): (sv_type, multiplicity, region)
                for sv_type, multiplicity, region in combinations
            }
            
            # Process results as they complete
            with tqdm(total=len(combinations), desc="Processing combinations", unit="combination") as pbar:
                for future in as_completed(future_to_combination):
                    sv_type, multiplicity, region = future_to_combination[future]
                    
                    try:
                        result = future.result()
                        
                        # Store result
                        all_results[sv_type][multiplicity['name']][region['name']] = result
                        
                        # Update summary
                        if result['success'] and result.get('blinded', False):
                            self.analysis_summary['blinded_combinations'] += 1
                        elif result['success']:
                            self.analysis_summary['successful_combinations'] += 1
                        else:
                            self.analysis_summary['failed_combinations'] += 1
                            
                        pbar.set_description(f"✓ {sv_type}/{multiplicity['name']}/{region['name']}")
                        
                    except Exception as exc:
                        print(f"\n✗ {sv_type}/{multiplicity['name']}/{region['name']} generated exception: {exc}")
                        all_results[sv_type][multiplicity['name']][region['name']] = {
                            'success': False, 'error': str(exc)
                        }
                        self.analysis_summary['failed_combinations'] += 1
                    
                    pbar.update(1)
    
    def print_analysis_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "="*80)
        print("MULTI-REGION SV ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 ANALYSIS OVERVIEW:")
        print("-" * 50)
        print(f"    • Total combinations: {self.analysis_summary['total_combinations']}")
        print(f"    • Successful: {self.analysis_summary['successful_combinations']}")
        print(f"    • Failed: {self.analysis_summary['failed_combinations']}")
        print(f"    • Blinded (signal regions): {self.analysis_summary['blinded_combinations']}")
        print(f"    • Files processed: {self.analysis_summary['files_processed']}")
        print(f"    • Total plots created: {self.analysis_summary['plots_created']}")
        
        print(f"\n🚀 PERFORMANCE:")
        print("-" * 50)
        print(f"    • Parallel processing: {'Enabled' if self.parallel_processing else 'Disabled'}")
        if self.parallel_processing:
            print(f"    • Number of workers: {self.n_processes}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)

def parse_mc_arguments(mc_args: List[str]) -> Tuple[List[str], List[float]]:
    """
    Parse MC arguments to separate file paths and optional scaling factors.
    
    Args:
        mc_args: List of arguments from --mc (mix of files and scaling factors)
        
    Returns:
        Tuple of (mc_files, scaling_factors) where scaling_factors[i] corresponds to mc_files[i]
        
    Example:
        Input: ['QCD.root', '12.0', 'WJets.root', 'ZToNuNu.root']
        Output: (['QCD.root', 'WJets.root', 'ZToNuNu.root'], [12.0, 1.0, 1.0])
    """
    mc_files = []
    scaling_factors = []
    
    i = 0
    while i < len(mc_args):
        arg = mc_args[i]
        
        # Check if this argument is a ROOT file
        if arg.endswith('.root'):
            mc_files.append(arg)
            
            # Check if the next argument is a scaling factor
            if i + 1 < len(mc_args):
                next_arg = mc_args[i + 1]
                try:
                    # Try to parse as float
                    scale_factor = float(next_arg)
                    scaling_factors.append(scale_factor)
                    i += 2  # Skip both the file and the scaling factor
                except ValueError:
                    # Next argument is not a number, so no scaling factor provided
                    scaling_factors.append(1.0)  # Default scaling
                    i += 1
            else:
                # No more arguments, so no scaling factor
                scaling_factors.append(1.0)
                i += 1
        else:
            # This shouldn't happen if the format is correct, but skip it
            print(f"Warning: Unexpected argument '{arg}' in MC list (expected ROOT file)")
            i += 1
    
    if len(mc_files) != len(scaling_factors):
        raise ValueError("Mismatch between MC files and scaling factors")
    
    return mc_files, scaling_factors

def main():
    parser = argparse.ArgumentParser(description='Multi-region SV variables analysis with comprehensive plotting')
    
    # Input files
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Data ROOT files')
    parser.add_argument('--mc', type=str, nargs='+', required=True,
                       help='MC background ROOT files with optional scaling factors (e.g., QCD.root 12.0 WJets.root ZToNuNu.root)')
    
    # Analysis configuration
    parser.add_argument('--luminosity', type=float, default=400.0,
                       help='Integrated luminosity in fb-1 (default: 400)')
    parser.add_argument('--variables', type=str, nargs='*',
                       help='Specific variables to plot (default: all)')
    parser.add_argument('--extra-cuts', type=str,
                       help='Additional cuts as string expression (e.g., "selCMet > 200 && rjr_Ms < 5000")')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize data and MC to their total integrals for shape comparison')
    
    # Output configuration
    parser.add_argument('--root-output', type=str, 
                       help='Output ROOT file name (skip ROOT output if not provided)')
    parser.add_argument('--pdf', action='store_true',
                       help='Save plots as PDF files in folder structure')
    parser.add_argument('--pdf-dir', type=str, default='sv_plots_output',
                       help='Base directory for PDF output (default: sv_plots_output)')
    
    # Performance options
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.root_output and not args.pdf:
        parser.error("Must specify either --root-output or --pdf (or both)")
    
    # Parse MC arguments to separate files and scaling factors
    mc_files, mc_scaling_factors = parse_mc_arguments(args.mc)
    
    # Print parsing results
    print("Parsed MC files and scaling factors:")
    for i, (mc_file, scale) in enumerate(zip(mc_files, mc_scaling_factors)):
        scale_info = f" (scale: {scale}x)" if scale != 1.0 else ""
        print(f"  {i+1}. {mc_file}{scale_info}")
    
    # Initialize analyzer
    analyzer = MultiRegionSVAnalyzer(
        luminosity=args.luminosity,
        parallel_processing=not args.no_parallel,
        n_processes=args.workers
    )
    
    # Run analysis
    analyzer.run_analysis(
        data_files=args.data,
        mc_files=mc_files,
        mc_scaling_factors=mc_scaling_factors,
        variables_to_plot=args.variables,
        output_root_file=args.root_output,
        pdf_output=args.pdf,
        pdf_base_dir=args.pdf_dir,
        extra_cuts=args.extra_cuts,
        normalize=args.normalize
    )

if __name__ == "__main__":
    try:
        main()
    finally:
        # Force cleanup to prevent hanging at exit
        import gc
        gc.collect()
        ROOT.gROOT.GetListOfCanvases().Clear()
        ROOT.gROOT.GetListOfFiles().Clear()
        # Force exit if still hanging
        import sys
        sys.exit(0)
