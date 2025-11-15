#!/usr/bin/env python3
"""
UnrolledDataProcessor - Pure data processing logic for unrolled yield analysis.

This class handles all the computational aspects:
- Loading ROOT files and applying cuts
- Calculating 2D yield matrices 
- Converting 2D yields to 1D unrolled format
- Managing bin definitions and labels

Extracted from unrolled_yield_analysis.py for modularity and reusability.
"""

import uproot
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FileData:
    """Container for loaded and filtered ROOT file data."""
    ms_values: np.ndarray
    rs_values: np.ndarray
    weights: np.ndarray
    baseline_mask: np.ndarray
    file_type: str
    file_path: str
    n_events_total: int
    n_events_after_baseline: int
    
    def apply_final_state_cut(self, final_state_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply additional final state cuts and return filtered data."""
        combined_mask = self.baseline_mask & final_state_mask
        return (
            self.ms_values[combined_mask],
            self.rs_values[combined_mask], 
            self.weights[combined_mask]
        )

class UnrolledDataProcessor:
    def __init__(self, luminosity: float = 400.0):
        """
        Initialize the data processor.
        
        Args:
            luminosity: Integrated luminosity in fb-1 (default 400 fb-1)
        """
        self.luminosity = luminosity
        
        # Define bin edges for Ms and Rs (same as yield_analysis.py)
        self.ms_bins = [1000, 2000, 3000, float('inf')]
        self.rs_bins = [0.15, 0.3, 0.4, float('inf')]
        
        # Define bracket-style labels with "inf" for infinity
        self.ms_bracket_labels = ["[1.0,2.0]", "[2.0,3.0]", "[3.0,inf]"]
        self.rs_bracket_labels = ["[0.15,0.3]", "[0.3,0.4]", "[0.4,inf]"]
        
        # Define group labels using #in notation with "inf"
        self.ms_group_labels = ["M_{S} #in[1000,2000]", "M_{S} #in[2000,3000]", "M_{S} #in[3000,inf]"]
        self.rs_group_labels = ["R_{S} #in[0.15,0.3]", "R_{S} #in[0.3,0.4]", "R_{S} #in[0.4,inf]"]
        
        # Processing summary for debugging
        self.processing_summary = {
            'files_processed': [],
            'baseline_cuts_applied': [],
            'final_state_selections_used': [],
            'errors': []
        }
    
    def apply_baseline_selection(self, tree, file_type: str, n_events: int, 
                                baseline_cuts: List[str] = None) -> np.ndarray:
        """
        Apply baseline selection cuts.
        
        Args:
            tree: ROOT tree object
            file_type: 'signal', 'background', or 'data'
            n_events: Number of events
            baseline_cuts: List of baseline cuts to apply (default: ['selCMet>150', 'hlt_flags', 'cleaning_flags', 'rjrCleaningVeto0'])
            
        Returns:
            Boolean mask with baseline cuts applied
        """
        # Default baseline cuts
        if baseline_cuts is None:
            baseline_cuts = ['selCMet>150', 'hlt_flags', 'cleaning_flags', 'rjrCleaningVeto0']
        
        # Initialize mask
        baseline_mask = np.ones(n_events, dtype=bool)
        
        for cut in baseline_cuts:
            if cut == 'selCMet>150':
                baseline_mask &= self._apply_selCMet_cut(tree, n_events)
            elif cut == 'hlt_flags':
                baseline_mask &= self._apply_hlt_flags_cut(tree, n_events)
            elif cut == 'cleaning_flags':
                baseline_mask &= self._apply_cleaning_flags_cut(tree, n_events)
            elif cut == 'rjrCleaningVeto0':
                baseline_mask &= self._apply_rjr_cleaning_veto_cut(tree, n_events)
            else:
                # Handle custom cuts in format "branch>value", "branch==value", etc.
                baseline_mask &= self._apply_custom_cut(tree, cut, n_events)
        
        return baseline_mask
    
    def _apply_selCMet_cut(self, tree, n_events: int) -> np.ndarray:
        """Apply selCMet > 150 cut."""
        try:
            selCMet_array = tree['selCMet'].array(library='np')
            if hasattr(selCMet_array, '__len__'):
                if hasattr(selCMet_array[0], '__len__'):
                    # If jagged, take first element
                    selCMet_values = np.array([event[0] if len(event) > 0 else 0 for event in selCMet_array])
                else:
                    selCMet_values = selCMet_array
                mask = (selCMet_values > 150)
                self.processing_summary['baseline_cuts_applied'].append('selCMet > 150')
                return mask
            else:
                self.processing_summary['errors'].append('selCMet has unexpected format')
                return np.zeros(n_events, dtype=bool)
        except uproot.KeyInFileError:
            self.processing_summary['errors'].append('selCMet branch not found')
            return np.zeros(n_events, dtype=bool)
    
    def _apply_hlt_flags_cut(self, tree, n_events: int) -> np.ndarray:
        """Apply hlt_flags cut."""
        try:
            trigger_array = tree['hlt_flags'].array(library='np')
            if hasattr(trigger_array, '__len__'):
                mask = trigger_array.astype(bool)
                self.processing_summary['baseline_cuts_applied'].append('hlt_flags')
                return mask
            else:
                self.processing_summary['errors'].append('hlt_flags has unexpected format')
                return np.zeros(n_events, dtype=bool)
        except uproot.KeyInFileError:
            self.processing_summary['errors'].append('hlt_flags branch not found')
            return np.zeros(n_events, dtype=bool)
    
    def _apply_cleaning_flags_cut(self, tree, n_events: int) -> np.ndarray:
        """Apply cleaning_flags cut."""
        # Check for common cleaning flag branches
        cleaning_branches = ['Flag_MetFilters', 'cleaning_flags', 'cleaningFlags', 'metFilters']
        for branch_name in cleaning_branches:
            try:
                cleaning_array = tree[branch_name].array(library='np')
                if hasattr(cleaning_array, '__len__'):
                    mask = cleaning_array.astype(bool)
                    self.processing_summary['baseline_cuts_applied'].append(f'{branch_name} (cleaning_flags)')
                    return mask
            except uproot.KeyInFileError:
                continue
        
        # If no cleaning flags found
        self.processing_summary['errors'].append('No cleaning flags branch found')
        return np.ones(n_events, dtype=bool)  # Return all true if no cleaning flags
    
    def _apply_rjr_cleaning_veto_cut(self, tree, n_events: int) -> np.ndarray:
        """Apply rjrCleaningVeto0 cut."""
        try:
            rjr_veto_array = tree['rjrCleaningVeto0'].array(library='np')
            if hasattr(rjr_veto_array, '__len__'):
                # Handle jagged arrays like other cuts
                if hasattr(rjr_veto_array[0], '__len__'):
                    # If jagged, take first element or 1 (fail) if empty
                    rjr_veto_values = np.array([event[0] if len(event) > 0 else 1 for event in rjr_veto_array])
                else:
                    rjr_veto_values = rjr_veto_array
                
                # The veto should be 0 (pass), so we want NOT veto (i.e., veto == 0)
                mask = (rjr_veto_values == 0)
                self.processing_summary['baseline_cuts_applied'].append('rjrCleaningVeto0')
                return mask
            else:
                self.processing_summary['errors'].append('rjrCleaningVeto0 has unexpected format')
                return np.zeros(n_events, dtype=bool)
        except uproot.KeyInFileError:
            self.processing_summary['errors'].append('rjrCleaningVeto0 branch not found - cut skipped')
            return np.ones(n_events, dtype=bool)  # Return all true if branch not found
    
    def _apply_custom_cut(self, tree, cut_str: str, n_events: int) -> np.ndarray:
        """Apply custom cut in format 'branch>value', 'branch==value', etc."""
        # Parse cut string
        for operator in ['>=', '<=', '==', '>', '<']:
            if operator in cut_str:
                parts = cut_str.split(operator)
                if len(parts) == 2:
                    branch_name = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                        branch_data = tree[branch_name].array(library='np')
                        
                        # Handle jagged arrays
                        if hasattr(branch_data[0], '__len__'):
                            values = np.array([event[0] if len(event) > 0 else 0 for event in branch_data])
                        else:
                            values = branch_data
                        
                        # Apply operator
                        if operator == '>':
                            mask = (values > value)
                        elif operator == '<':
                            mask = (values < value)
                        elif operator == '>=':
                            mask = (values >= value)
                        elif operator == '<=':
                            mask = (values <= value)
                        elif operator == '==':
                            mask = (values == value)
                        
                        self.processing_summary['baseline_cuts_applied'].append(cut_str)
                        return mask
                        
                    except (ValueError, uproot.KeyInFileError):
                        self.processing_summary['errors'].append(f'Failed to apply cut: {cut_str}')
                        return np.ones(n_events, dtype=bool)
                break
        
        self.processing_summary['errors'].append(f'Unable to parse cut: {cut_str}')
        return np.ones(n_events, dtype=bool)
    
    def apply_final_state_selection(self, tree, final_state_flags: List[str], n_events: int) -> np.ndarray:
        """
        Apply final state selection flags.
        
        Args:
            tree: ROOT tree object
            final_state_flags: List of final state selection flag names
            n_events: Number of events
            
        Returns:
            Boolean mask with final state selections applied (OR of all flags)
        """
        if not final_state_flags:
            return np.ones(n_events, dtype=bool)
        
        combined_mask = np.zeros(n_events, dtype=bool)
        
        for flag in final_state_flags:
            try:
                selection_array = tree[flag].array(library='np')
                if hasattr(selection_array, '__len__'):
                    flag_mask = selection_array.astype(bool)
                    combined_mask |= flag_mask  # OR combination
                    self.processing_summary['final_state_selections_used'].append(flag)
                else:
                    self.processing_summary['errors'].append(f'{flag} has unexpected format')
            except uproot.KeyInFileError:
                self.processing_summary['errors'].append(f'{flag} branch not found')
        
        return combined_mask
    
    def load_file_data(self, file_path: str, file_type: str,
                      baseline_cuts: List[str] = None,
                      signal_scale: float = 1.0,
                      background_scale: float = 1.0) -> FileData:
        """
        Load and cache data from a ROOT file with baseline cuts applied.
        
        This method opens the file once, reads all required branches, applies
        baseline cuts, and returns a FileData object that can be reused for
        multiple final state selections.
        
        Args:
            file_path: Path to ROOT file
            file_type: 'signal', 'background', or 'data'
            baseline_cuts: List of baseline cuts (default: ['selCMet>150', 'hlt_flags', 'cleaning_flags'])
            signal_scale: Scaling factor for signal
            background_scale: Scaling factor for background
            
        Returns:
            FileData object containing pre-processed data
        """
        # Default baseline cuts
        if baseline_cuts is None:
            baseline_cuts = ['selCMet>150', 'hlt_flags', 'cleaning_flags', 'rjrCleaningVeto0']
        
        # Open file with uproot
        with uproot.open(file_path) as file:
            # Find the tree
            tree_name = None
            for key_name in file.keys():
                if file[key_name].classname == "TTree":
                    tree_name = key_name
                    break
            
            if not tree_name:
                raise ValueError(f"No TTree found in {file_path}")
            
            tree = file[tree_name]
            
            # Get required branches
            try:
                ms_array = tree['rjr_Ms'].array(library='np')
                rs_array = tree['rjr_Rs'].array(library='np')
                
                # Handle jagged arrays
                if hasattr(ms_array, '__len__') and len(ms_array) > 0:
                    if hasattr(ms_array[0], '__len__'):
                        # If jagged, extract index 0 from each event
                        ms_values = np.array([event[0] if len(event) > 0 else 0 for event in ms_array])
                        rs_values = np.array([event[0] if len(event) > 0 else 0 for event in rs_array])
                    else:
                        ms_values = ms_array
                        rs_values = rs_array
                else:
                    raise ValueError("Empty arrays found")
                
                n_events = len(ms_values)
                
                # Apply baseline selection
                baseline_mask = self.apply_baseline_selection(tree, file_type, n_events, baseline_cuts)
                
                # Get weights
                if file_type == 'data':
                    weights = np.ones(n_events)
                else:
                    evt_weights = tree['evtFillWgt'].array(library='np')
                    weights = evt_weights * self.luminosity
                    if file_type == 'signal':
                        weights *= signal_scale
                    elif file_type == 'background':
                        weights *= background_scale
                
            except uproot.KeyInFileError as e:
                raise ValueError(f"Required branch not found in {file_path}: {e}")
        
        # Record processing info
        self.processing_summary['files_processed'].append(f"{file_type}: {file_path}")
        
        return FileData(
            ms_values=ms_values,
            rs_values=rs_values,
            weights=weights,
            baseline_mask=baseline_mask,
            file_type=file_type,
            file_path=file_path,
            n_events_total=n_events,
            n_events_after_baseline=np.sum(baseline_mask)
        )
    
    def calculate_2d_yields_from_data(self, file_data: FileData,
                                     final_state_flags: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate 2D yield matrix from pre-loaded FileData.
        
        Args:
            file_data: Pre-loaded FileData object
            final_state_flags: Final state selection flags
            
        Returns:
            Tuple of (yields_2d, errors_2d) as 2D numpy arrays
        """
        # Initialize yield matrix (3x3 for Ms x Rs bins)
        yields = np.zeros((3, 3))
        sum_weights_squared = np.zeros((3, 3))
        
        if final_state_flags:
            # We need to load final state branches from the original file
            # This is a limitation - we'll need to open the file again for final state cuts
            # TODO: Could be optimized further by caching all potential final state branches
            with uproot.open(file_data.file_path) as file:
                tree_name = None
                for key_name in file.keys():
                    if file[key_name].classname == "TTree":
                        tree_name = key_name
                        break
                tree = file[tree_name]
                
                final_state_mask = self.apply_final_state_selection(tree, final_state_flags, file_data.n_events_total)
            
            # Apply both baseline and final state cuts
            ms_values, rs_values, weights = file_data.apply_final_state_cut(final_state_mask)
        else:
            # Apply only baseline cuts
            ms_values = file_data.ms_values[file_data.baseline_mask]
            rs_values = file_data.rs_values[file_data.baseline_mask]
            weights = file_data.weights[file_data.baseline_mask]
        
        # Calculate bin indices
        ms_bins = np.digitize(ms_values, self.ms_bins) - 1
        rs_bins = np.digitize(rs_values, self.rs_bins) - 1
        
        # Fill yield matrix
        for i in range(len(ms_values)):
            ms_bin = ms_bins[i]
            rs_bin = rs_bins[i]
            
            # Check if within valid bins (0, 1, 2 for each axis)
            if 0 <= ms_bin < 3 and 0 <= rs_bin < 3:
                yields[ms_bin, rs_bin] += weights[i]
                sum_weights_squared[ms_bin, rs_bin] += weights[i]**2
        
        # Calculate errors
        errors = np.sqrt(sum_weights_squared)
        
        return yields, errors
    
    def calculate_2d_yields(self, file_path: str, file_type: str, 
                           baseline_cuts: List[str] = None,
                           final_state_flags: List[str] = None,
                           signal_scale: float = 1.0, 
                           background_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate 2D yield matrix from ROOT file.
        
        Args:
            file_path: Path to ROOT file
            file_type: 'signal', 'background', or 'data'
            baseline_cuts: List of baseline cuts to apply
            final_state_flags: List of final state selection flags
            signal_scale: Scaling factor for signal
            background_scale: Scaling factor for background
            
        Returns:
            Tuple of (yields_2d, errors_2d) as 3x3 numpy arrays
        """
        # Initialize yield matrix (3x3 for Ms x Rs bins)
        yields = np.zeros((3, 3))
        sum_weights_squared = np.zeros((3, 3))
        
        # Open file with uproot
        with uproot.open(file_path) as file:
            # Find the tree
            tree_name = None
            for key_name in file.keys():
                if file[key_name].classname == "TTree":
                    tree_name = key_name
                    break
            
            if not tree_name:
                raise ValueError(f"No TTree found in {file_path}")
            
            tree = file[tree_name]
            
            # Get required branches
            try:
                ms_array = tree['rjr_Ms'].array(library='np')
                rs_array = tree['rjr_Rs'].array(library='np')
                
                # Check if arrays are jagged or regular
                if hasattr(ms_array, '__len__') and len(ms_array) > 0:
                    # If jagged arrays, extract index 0 from each event
                    if hasattr(ms_array[0], '__len__'):
                        ms_values = np.array([event[0] if len(event) > 0 else 0 for event in ms_array])
                        rs_values = np.array([event[0] if len(event) > 0 else 0 for event in rs_array])
                    else:
                        ms_values = ms_array
                        rs_values = rs_array
                else:
                    raise ValueError("Empty arrays found")
                
                n_events = len(ms_values)
                
                # Apply baseline selection
                baseline_mask = self.apply_baseline_selection(tree, file_type, n_events, baseline_cuts)
                
                # Apply final state selection if specified
                if final_state_flags:
                    final_state_mask = self.apply_final_state_selection(tree, final_state_flags, n_events)
                    combined_mask = baseline_mask & final_state_mask
                else:
                    combined_mask = baseline_mask
                
                # Apply final filtering
                ms_values = ms_values[combined_mask]
                rs_values = rs_values[combined_mask]
                
                # Get weights after filtering
                if file_type == 'data':
                    weights = np.ones(len(ms_values))
                else:
                    evt_weights = tree['evtFillWgt'].array(library='np')[combined_mask]
                    weights = evt_weights * self.luminosity
                    if file_type == 'signal':
                        weights *= signal_scale
                    elif file_type == 'background':
                        weights *= background_scale
                
            except uproot.KeyInFileError as e:
                raise ValueError(f"Required branch not found in {file_path}: {e}")
        
        # Calculate bin indices for all events
        ms_bins = np.digitize(ms_values, self.ms_bins) - 1
        rs_bins = np.digitize(rs_values, self.rs_bins) - 1
        
        # Fill yield matrix
        for i in range(len(ms_values)):
            ms_bin = ms_bins[i]
            rs_bin = rs_bins[i]
            
            # Check if within valid bins (0, 1, 2 for each axis)
            if 0 <= ms_bin < 3 and 0 <= rs_bin < 3:
                yields[ms_bin, rs_bin] += weights[i]
                sum_weights_squared[ms_bin, rs_bin] += weights[i]**2
        
        # Calculate errors
        errors = np.sqrt(sum_weights_squared)
        
        # Update processing summary
        self.processing_summary['files_processed'].append(f"{file_type}: {file_path}")
        
        return yields, errors
    
    def unroll_2d_to_1d_grouped_by_ms(self, yields_2d: np.ndarray, errors_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Unroll 2D yields to 1D histogram grouped by Ms.
        
        Bin structure: [Ms1_Rs1, Ms1_Rs2, Ms1_Rs3, Ms2_Rs1, Ms2_Rs2, Ms2_Rs3, Ms3_Rs1, Ms3_Rs2, Ms3_Rs3]
        
        Args:
            yields_2d: 3x3 yield matrix
            errors_2d: 3x3 error matrix
            
        Returns:
            Tuple of (yields_1d, errors_1d, bin_labels)
        """
        yields_1d = np.zeros(9)
        errors_1d = np.zeros(9)
        bin_labels = []
        
        bin_idx = 0
        for ms_idx in range(3):
            for rs_idx in range(3):
                yields_1d[bin_idx] = yields_2d[ms_idx, rs_idx]
                errors_1d[bin_idx] = errors_2d[ms_idx, rs_idx]
                bin_labels.append(self.rs_bracket_labels[rs_idx])  # Show Rs labels for Ms grouping
                bin_idx += 1
        
        return yields_1d, errors_1d, bin_labels
    
    def unroll_2d_to_1d_grouped_by_rs(self, yields_2d: np.ndarray, errors_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Unroll 2D yields to 1D histogram grouped by Rs.
        
        Bin structure: [Rs1_Ms1, Rs1_Ms2, Rs1_Ms3, Rs2_Ms1, Rs2_Ms2, Rs2_Ms3, Rs3_Ms1, Rs3_Ms2, Rs3_Ms3]
        
        Args:
            yields_2d: 3x3 yield matrix
            errors_2d: 3x3 error matrix
            
        Returns:
            Tuple of (yields_1d, errors_1d, bin_labels)
        """
        yields_1d = np.zeros(9)
        errors_1d = np.zeros(9)
        bin_labels = []
        
        bin_idx = 0
        for rs_idx in range(3):
            for ms_idx in range(3):
                yields_1d[bin_idx] = yields_2d[ms_idx, rs_idx]
                errors_1d[bin_idx] = errors_2d[ms_idx, rs_idx]
                bin_labels.append(self.ms_bracket_labels[ms_idx])  # Show Ms labels for Rs grouping
                bin_idx += 1
        
        return yields_1d, errors_1d, bin_labels
    
    def unroll_2d_to_1d(self, yields_2d: np.ndarray, errors_2d: np.ndarray, 
                       grouping_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Unroll 2D yields to 1D histogram with specified grouping.
        
        Args:
            yields_2d: 3x3 yield matrix
            errors_2d: 3x3 error matrix
            grouping_type: 'ms' or 'rs' for grouping type
            
        Returns:
            Tuple of (yields_1d, errors_1d, bin_labels)
        """
        if grouping_type == 'ms':
            return self.unroll_2d_to_1d_grouped_by_ms(yields_2d, errors_2d)
        elif grouping_type == 'rs':
            return self.unroll_2d_to_1d_grouped_by_rs(yields_2d, errors_2d)
        else:
            raise ValueError(f"Unknown grouping_type: {grouping_type}. Must be 'ms' or 'rs'")
    
    def get_group_labels(self, grouping_type: str) -> List[str]:
        """
        Get group labels for the specified grouping type.
        
        Args:
            grouping_type: 'ms' or 'rs' for grouping type
            
        Returns:
            List of group label strings
        """
        if grouping_type == 'ms':
            return self.ms_group_labels
        elif grouping_type == 'rs':
            return self.rs_group_labels
        else:
            raise ValueError(f"Unknown grouping_type: {grouping_type}. Must be 'ms' or 'rs'")
    
    def get_axis_title(self, grouping_type: str) -> str:
        """
        Get the appropriate x-axis title for the grouping type.
        
        Args:
            grouping_type: 'ms' or 'rs' for grouping type
            
        Returns:
            X-axis title string
        """
        return "R_{S}" if grouping_type == 'ms' else "M_{S}"
    
    def print_processing_summary(self) -> None:
        """Print summary of data processing operations."""
        print("\n" + "="*60)
        print("DATA PROCESSING SUMMARY")
        print("="*60)
        
        if self.processing_summary['files_processed']:
            print(f"\n📁 FILES PROCESSED ({len(self.processing_summary['files_processed'])}):")
            print("-" * 40)
            for file_info in self.processing_summary['files_processed']:
                print(f"    • {file_info}")
        
        if self.processing_summary['baseline_cuts_applied']:
            print(f"\n✂️  BASELINE CUTS APPLIED:")
            print("-" * 40)
            for cut in set(self.processing_summary['baseline_cuts_applied']):
                print(f"    • {cut}")
        
        if self.processing_summary['final_state_selections_used']:
            print(f"\n🎯 FINAL STATE SELECTIONS USED:")
            print("-" * 40)
            for selection in set(self.processing_summary['final_state_selections_used']):
                print(f"    • {selection}")
        
        if self.processing_summary['errors']:
            print(f"\n⚠️  WARNINGS/ERRORS:")
            print("-" * 40)
            for error in set(self.processing_summary['errors']):
                print(f"    • {error}")
        
        print("\n" + "="*60)
