#!/usr/bin/env python3
"""
SV Variables Plotting Utilities

Modular utilities for creating Data/MC comparison plots of SV variables.
Designed to integrate with multi-region analysis frameworks.
"""

import ROOT
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import uproot

# Enable batch mode and disable title/stat boxes
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0)

class SVVariablesPlotter:
    """Core class for creating SV variable comparison plots."""
    
    def __init__(self, luminosity: float = 400.0):
        """
        Initialize the SV variables plotter.
        
        Args:
            luminosity: Integrated luminosity in fb-1 for MC scaling
        """
        self.luminosity = luminosity
        
        # Define variable configurations by SV type: (bins, x_label, log_scale)
        self.hadronic_variables = {
            'HadronicSV_mass': (np.linspace(0, 100, 26), 'mass [GeV]', True),
            'HadronicSV_dxy': (np.linspace(0, 50, 51), 'd_{xy} [cm]', True),
            'HadronicSV_pOverE': (np.linspace(0.6, 1, 26), 'p/E', True),
            'HadronicSV_decayAngle': (np.linspace(-1, 1, 26), 'cos#theta_{CM}^{*}', True),
            'HadronicSV_cosTheta': (np.linspace(0., 1, 26), 'cos#theta', True),
            'HadronicSV_dxySig': (np.linspace(0, 1000, 26), 'S_{xy}', True),
            'rjr_Ms_0': (np.linspace(0, 8000, 51), 'M_{S} [GeV]', True),
            'rjr_Rs_0': (np.linspace(0, 1, 51), 'R_{S}', True),
            'selCMet': (np.linspace(0, 1000, 51), 'p_{T}^{miss} [GeV]', True)
        }
        
        self.leptonic_variables = {
            'LeptonicSV_mass': (np.linspace(0, 100, 26), 'mass [GeV]', True),
            'LeptonicSV_dxy': (np.linspace(0, 50, 51), 'd_{xy} [cm]', True),
            'LeptonicSV_pOverE': (np.linspace(0.6, 1, 26), 'p/E', True),
            'LeptonicSV_decayAngle': (np.linspace(-1, 1, 26), 'cos#theta_{CM}^{*}', True),
            'LeptonicSV_cosTheta': (np.linspace(0.75, 1, 26), 'cos#theta', True),
            'LeptonicSV_dxySig': (np.linspace(0, 1000, 26), 'S_{xy}', True),
            'rjr_Ms_0': (np.linspace(0, 8000, 51), 'M_{S} [GeV]', True),
            'rjr_Rs_0': (np.linspace(0, 1, 51), 'R_{S}', True),
            'selCMet': (np.linspace(0, 1000, 51), 'p_{T}^{miss} [GeV]', True)
        }
        
        # For AnySV, use both types (we'll need to handle this case)
        self.any_variables = {
            **self.hadronic_variables,
            **self.leptonic_variables
        }
        
        # For 0-SV, only event-level variables (no SV-specific variables)
        self.zero_sv_variables = {
            'rjr_Ms_0': (np.linspace(0, 8000, 51), 'M_{S} [GeV]', True),
            'rjr_Rs_0': (np.linspace(0, 1, 51), 'R_{S}', True),
            'selCMet': (np.linspace(0, 1000, 51), 'MET [GeV]', True)
        }
        
        # Backward compatibility
        self.variables = self.hadronic_variables
        
        # Colors and styles following ROOT conventions
        self.data_color = ROOT.kBlack
        self.mc_color = ROOT.kRed+2
        self.ratio_color = ROOT.kBlack
        
        # Create stable colors (fixed ROOT color persistence issues)
        self.mc_colors = self._create_stable_colors()
    
    def get_variables_for_sv_type(self, sv_type: str) -> dict:
        """Get the appropriate variables for the given SV type."""
        if sv_type == 'HadronSV':
            return self.hadronic_variables
        elif sv_type == 'LeptonSV':
            return self.leptonic_variables
        elif sv_type == 'AnySV':
            return self.any_variables
        elif sv_type == '0SV':
            return self.zero_sv_variables
        else:
            # Default to hadronic for backward compatibility
            return self.hadronic_variables
    
    def _create_stable_colors(self) -> List[int]:
        """Create stable ROOT colors that persist in saved files."""
        mc_colors = []
        hex_colors = ["#5A4484", "#347889", "#F4B240", "#E54B26", "#C05780", "#7A68A6", "#2E8B57", "#8B4513"]
        
        for i, hex_color in enumerate(hex_colors):
            # Convert hex to RGB (0-255 range for ROOT)
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16) 
            b = int(hex_color[4:6], 16)
            
            # Create color using RGB values directly - same method as working dxySig script
            color_index = ROOT.TColor.GetColor(r, g, b)
            mc_colors.append(color_index)
        
        return mc_colors
    
    def _clean_mc_label(self, label: str) -> str:
        """Extract clean physics process name from file-based label."""
        # Remove common suffixes
        clean_label = label.replace("Skim_v43", "").replace("Skim", "").replace("_v43", "")
        
        # Map to standard physics process names
        label_mapping = {
            'QCD': 'QCD multijets',
            'WJets': 'W + jets', 
            'ZJets': 'Z + jets',
            'GJets': '#gamma + jets',
            'TTXJets': 't#bar{t} + X',
            'TTJets': 't#bar{t} + jets'
        }
        
        # Find matching process
        for key, clean_name in label_mapping.items():
            if key in clean_label:
                return clean_name
        
        # Fallback to cleaned label if no mapping found
        return clean_label.strip('_')
    
    def load_branches_for_sv_plotting(self, file_path: str) -> Dict:
        """Load all branches needed for SV variable plotting."""
        try:
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
                
                # Load required branches for SV variable plotting
                branches_to_load = [
                    'evtFillWgt',
                    'SV_nLeptonic', 'SV_nHadronic',
                    'HadronicSV_dxySig', 'HadronicSV_mass', 'HadronicSV_dxy',
                    'HadronicSV_pOverE', 'HadronicSV_decayAngle', 'HadronicSV_cosTheta',
                    'LeptonicSV_dxySig', 'LeptonicSV_mass', 'LeptonicSV_dxy',
                    'LeptonicSV_pOverE', 'LeptonicSV_decayAngle', 'LeptonicSV_cosTheta',
                    'rjr_Ms', 'rjr_Rs', 'selCMet', 'hlt_flags', 'Flag_MetFilters', 'rjrPTS'
                ]
                
                data = {}
                for branch in branches_to_load:
                    try:
                        array_data = tree[branch].array(library='np')
                        # Handle jagged arrays for rjr_Ms and rjr_Rs
                        if branch in ['rjr_Ms', 'rjr_Rs'] and hasattr(array_data[0], '__len__'):
                            # Extract the [0] element from each event, defaulting to 0 if not available
                            data[branch] = np.array([event[0] if len(event) > 0 else 0 for event in array_data])
                        else:
                            data[branch] = array_data
                    except uproot.KeyInFileError:
                        print(f"Warning: Branch '{branch}' not found in {file_path}")
                        data[branch] = None
                
                return data
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}

class RegionSelector:
    """Handles event selection logic for different regions and multiplicities."""
    
    def __init__(self):
        pass
    
    def apply_region_selection(self, file_data: Dict, region_config: Dict, 
                             multiplicity_config: Dict, sv_type: str = 'HadronSV',
                             extra_cuts: Optional[str] = None, file_type: str = 'data') -> np.ndarray:
        """
        Apply region selection cuts and return boolean mask.
        
        Args:
            file_data: Loaded ROOT file data
            region_config: Region configuration (dxySig cuts, etc.)
            multiplicity_config: Multiplicity configuration (nHad, nLep cuts)
            sv_type: SV type ('HadronSV', 'LeptonSV', or 'AnySV')
            extra_cuts: Optional string with additional cuts
            
        Returns:
            Boolean mask array indicating which events pass selection
        """
        n_events = len(file_data['SV_nLeptonic']) if file_data['SV_nLeptonic'] is not None else 0
        if n_events == 0:
            return np.array([], dtype=bool)
        
        # Start with all events
        mask = np.ones(n_events, dtype=bool)
        
        # Apply default cuts (hlt_flags, Flag_MetFilters, selCMet > 150, rjrPTS < 150)
        default_mask = self._apply_default_cuts(file_data, file_type, n_events)
        mask &= default_mask
        
        # Apply multiplicity cuts
        mult_mask = self._apply_multiplicity_cuts(file_data, multiplicity_config, n_events)
        mask &= mult_mask
        
        # Apply region-specific cuts (dxySig, etc.)
        region_mask = self._apply_region_cuts(file_data, region_config, n_events, sv_type)
        mask &= region_mask
        
        # Apply extra cuts if provided
        extra_mask = self._apply_extra_cuts(file_data, n_events, extra_cuts)
        
        # Apply extra cuts
        if extra_cuts:
            events_before_extra = np.sum(mask)
            mask &= extra_mask
            events_after_extra = np.sum(mask)
            #if events_before_extra > 0:  # Only print if there were events to begin with
            #print(f"Applied extra cuts: {events_before_extra} -> {events_after_extra} events")
        
        return mask
    
    def _apply_extra_cuts(self, file_data: Dict, n_events: int, extra_cuts: Optional[str]) -> np.ndarray:
        """Apply additional cuts from string expression."""
        mask = np.ones(n_events, dtype=bool)
        
        if not extra_cuts:
            return mask
            
        try:
            # Parse and evaluate the cut string
            # Replace && and || with & and |
            cut_expr = extra_cuts.replace('&&', ' & ').replace('||', ' | ')
            
            # Find all branch names in the expression
            import re
            branch_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            potential_branches = re.findall(branch_pattern, cut_expr)
            
            # Filter out operators and keywords
            reserved_words = {'and', 'or', 'not', 'True', 'False', 'None'}
            branch_names = [b for b in potential_branches if b not in reserved_words]
            
            # Create evaluation environment
            eval_env = {'__builtins__': {}}
            
            # Add branch data to environment
            for branch in branch_names:
                if branch in file_data and file_data[branch] is not None:
                    branch_data = file_data[branch]
                    # Ensure consistent data types for boolean operations
                    if hasattr(branch_data, 'dtype'):
                        # Convert integer arrays to float for safer arithmetic
                        if branch_data.dtype in [np.int32, np.int64, np.uint32, np.uint64]:
                            branch_data = branch_data.astype(np.float64)
                        # Keep boolean and float arrays as-is
                    eval_env[branch] = branch_data
                else:
                    print(f"Warning: Branch '{branch}' in extra cuts not found in data")
                    return np.zeros(n_events, dtype=bool)  # Fail safe - no events pass
            
            # Add numpy for mathematical functions
            eval_env['np'] = np
            
            # Optional debug for rjr_Ms (commented out to reduce verbosity)
            # if 'rjr_Ms' in branch_names and 'rjr_Ms' in eval_env:
            #     rjr_Ms_data = eval_env['rjr_Ms']
            #     print(f"rjr_Ms range: min={np.min(rjr_Ms_data):.1f}, max={np.max(rjr_Ms_data):.1f}")
            #     print(f"rjr_Ms > 1000: {np.sum(rjr_Ms_data > 1000)} / {len(rjr_Ms_data)} events")
            
            # Use step-by-step evaluation to avoid parsing issues
            result = self._evaluate_expression_safely(cut_expr, eval_env)
            
            # Convert to boolean array if needed
            if hasattr(result, '__len__') and len(result) == n_events:
                # Ensure the result is a boolean array
                if result.dtype != bool:
                    mask = result.astype(bool)
                else:
                    mask = result
            elif isinstance(result, (bool, np.bool_)):
                # Single boolean value - apply to all events
                mask = np.full(n_events, result, dtype=bool)
            else:
                print(f"Warning: Extra cut expression '{extra_cuts}' did not return valid boolean array")
        
        except Exception as e:
            print(f"Error evaluating extra cuts '{extra_cuts}': {str(e)}")
            print(f"Cut expression after parsing: '{cut_expr}'")
            # Show data types for debugging
            for branch in branch_names:
                if branch in eval_env:
                    data = eval_env[branch]
                    print(f"  {branch}: type={type(data)}, dtype={getattr(data, 'dtype', 'N/A')}, shape={getattr(data, 'shape', 'N/A')}")
            mask = np.zeros(n_events, dtype=bool)  # Fail safe
        
        return mask
    
    def _evaluate_expression_safely(self, cut_expr: str, eval_env: Dict) -> np.ndarray:
        """Safely evaluate boolean expressions by splitting on logical operators."""
        try:
            # First try direct evaluation
            return eval(cut_expr, eval_env)
        except:
            # If that fails, split on logical operators and evaluate step by step
            if ' & ' in cut_expr:
                parts = cut_expr.split(' & ')
                result = None
                for part in parts:
                    part_result = eval(part.strip(), eval_env)
                    part_result = np.asarray(part_result, dtype=bool)
                    if result is None:
                        result = part_result
                    else:
                        result = result & part_result
                return result
            elif ' | ' in cut_expr:
                parts = cut_expr.split(' | ')
                result = None
                for part in parts:
                    part_result = eval(part.strip(), eval_env)
                    part_result = np.asarray(part_result, dtype=bool)
                    if result is None:
                        result = part_result
                    else:
                        result = result | part_result
                return result
            else:
                # Single expression, re-raise the error
                raise
    
    def _apply_default_cuts(self, file_data: Dict, file_type: str, n_events: int) -> np.ndarray:
        """Apply default cuts (hlt_flags, Flag_MetFilters, selCMet > 150, rjrPTS < 150)."""
        mask = np.ones(n_events, dtype=bool)
        
        # Apply hlt_flags (all file types)
        if 'hlt_flags' in file_data and file_data['hlt_flags'] is not None:
            mask &= file_data['hlt_flags'].astype(bool)
        
        # Apply Flag_MetFilters (all file types)
        if 'Flag_MetFilters' in file_data and file_data['Flag_MetFilters'] is not None:
            mask &= file_data['Flag_MetFilters'].astype(bool)
        
        # Apply selCMet > 150 (all file types)
        if 'selCMet' in file_data and file_data['selCMet'] is not None:
            selcmet_data = file_data['selCMet']
            # Handle both scalar and jagged arrays
            if len(selcmet_data.shape) > 1 or hasattr(selcmet_data[0], '__len__'):
                # If jagged, take first element of each event
                selcmet_values = np.array([event[0] if len(event) > 0 else 0 for event in selcmet_data])
            else:
                selcmet_values = selcmet_data
            mask &= selcmet_values > 150
        
        # Apply rjrPTS < 150 (all file types)
        if 'rjrPTS' in file_data and file_data['rjrPTS'] is not None:
            rjrpts_data = file_data['rjrPTS']
            # Handle both scalar and jagged arrays
            if len(rjrpts_data.shape) > 1 or hasattr(rjrpts_data[0], '__len__'):
                # If jagged, take first element of each event
                rjrpts_values = np.array([event[0] if len(event) > 0 else 0 for event in rjrpts_data])
            else:
                rjrpts_values = rjrpts_data
            mask &= rjrpts_values < 150
        
        return mask
    
    def _apply_multiplicity_cuts(self, file_data: Dict, multiplicity_config: Dict, 
                               n_events: int) -> np.ndarray:
        """Apply multiplicity cuts (nHad==1, nLep==1, etc.)."""
        mask = np.ones(n_events, dtype=bool)
        
        for cut_str in multiplicity_config.get('cuts', []):
            parts = cut_str.split(':')
            if len(parts) != 3:
                continue
                
            branch, operator, value_str = parts
            try:
                value = float(value_str)
                branch_data = file_data.get(branch)
                if branch_data is None:
                    mask = np.zeros(n_events, dtype=bool)
                    break
                
                # Apply cut
                if operator == '==':
                    mask &= (branch_data == value)
                elif operator == '>':
                    mask &= (branch_data > value)
                elif operator == '>=':
                    mask &= (branch_data >= value)
                elif operator == '<':
                    mask &= (branch_data < value)
                elif operator == '<=':
                    mask &= (branch_data <= value)
                    
            except (ValueError, TypeError):
                continue
        
        return mask
    
    def _apply_region_cuts(self, file_data: Dict, region_config: Dict, 
                         n_events: int, sv_type: str = 'HadronSV') -> np.ndarray:
        """Apply region-specific cuts (dxySig ranges, etc.)."""
        mask = np.ones(n_events, dtype=bool)
        
        # Handle different region types
        if 'dxySig_cut' in region_config:
            mask &= self._apply_dxysig_cuts(file_data, region_config, n_events, sv_type)
        
        return mask
    
    def _apply_dxysig_cuts(self, file_data: Dict, region_config: Dict, 
                         n_events: int, sv_type: str = 'HadronSV') -> np.ndarray:
        """Apply dxySig cuts for region selection."""
        
        # For 0-SV categories, skip dxySig cuts entirely
        if sv_type == '0SV':
            return np.ones(n_events, dtype=bool)
        
        # For SV variable plotting, we typically want sideband regions (dxySig < 1000)
        # This is different from the full multi-region analysis
        
        # Default: require dxySig < 1000 (sideband) unless specified otherwise
        dxysig_cut = region_config.get('dxySig_cut', '<:1000')
        
        # Choose the appropriate dxySig branch based on SV type
        dxy_sig_data = None
        if sv_type == 'HadronSV':
            dxy_sig_data = file_data.get('HadronicSV_dxySig')
        elif sv_type == 'LeptonSV':
            dxy_sig_data = file_data.get('LeptonicSV_dxySig')
        elif sv_type == 'AnySV':
            # For AnySV, try hadronic first, then leptonic
            had_dxy_sig = file_data.get('HadronicSV_dxySig')
            lep_dxy_sig = file_data.get('LeptonicSV_dxySig')
            if had_dxy_sig is not None:
                dxy_sig_data = had_dxy_sig
            elif lep_dxy_sig is not None:
                dxy_sig_data = lep_dxy_sig
        
        if dxy_sig_data is None:
            return np.zeros(n_events, dtype=bool)
        
        # Handle array structure
        if hasattr(dxy_sig_data[0], '__len__'):
            # Take first element of each event's array
            dxy_values = np.array([event[0] if len(event) > 0 else 0 for event in dxy_sig_data])
        else:
            dxy_values = dxy_sig_data
        
        # Parse and apply cut - handle compound cuts like ">=:90,<:300"
        # Always require positive dxySig values first
        mask = (dxy_values > 0)
        
        if ',' in dxysig_cut:
            # Handle compound cuts (e.g., ">=:90,<:300")
            individual_cuts = dxysig_cut.split(',')
            for cut in individual_cuts:
                mask &= self._parse_single_dxysig_cut(cut, dxy_values)
        else:
            # Handle single cuts
            mask &= self._parse_single_dxysig_cut(dxysig_cut, dxy_values)
        
        return mask
    
    def _parse_single_dxysig_cut(self, cut_str: str, dxy_values: np.ndarray) -> np.ndarray:
        """Parse and apply a single dxySig cut."""
        if ':' not in cut_str:
            # Default fallback - assume sideband region
            return dxy_values < 1000
        
        operator, value_str = cut_str.split(':')
        value = float(value_str)
        
        if operator == '<':
            return dxy_values < value
        elif operator == '>':
            return dxy_values > value
        elif operator == '>=':
            return dxy_values >= value
        elif operator == '<=':
            return dxy_values <= value
        else:
            # Default fallback - assume sideband region
            return dxy_values < 1000
    
    def extract_variable_values(self, file_data: Dict, file_type: str, 
                              variable_name: str, event_mask: np.ndarray, luminosity: float = 400.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract variable values and weights for events passing selection.
        
        Args:
            file_data: Loaded ROOT file data
            file_type: 'data' or 'mc'
            variable_name: Name of variable to extract
            event_mask: Boolean mask of events passing selection
            
        Returns:
            Tuple of (values, weights) arrays
        """
        values = []
        weights = []
        
        n_events = len(event_mask)
        
        n_passed = 0
        for i in range(n_events):
            if not event_mask[i]:
                continue
            n_passed += 1
            
            # Get base event weight
            if file_type == 'data':
                base_event_weight = 1.0
            else:
                base_event_weight = file_data['evtFillWgt'][i] * luminosity if file_data['evtFillWgt'] is not None else 0.0
            
            # Extract the specific variable value
            var_value = None
            
            if variable_name.startswith('HadronicSV_') or variable_name.startswith('LeptonicSV_'):
                branch_name = variable_name
                if file_data[branch_name] is not None:
                    var_data = file_data[branch_name][i]
                    if hasattr(var_data, '__len__'):
                        if len(var_data) > 0:
                            var_value = var_data[0]  # Take first SV
                    else:
                        var_value = var_data
            
            elif variable_name == 'rjr_Ms_0':
                if file_data['rjr_Ms'] is not None:
                    rjr_ms = file_data['rjr_Ms'][i]
                    # rjr_Ms is already processed to extract [0] element during data loading
                    if hasattr(rjr_ms, '__len__') and len(rjr_ms) > 0:
                        var_value = rjr_ms[0]  # For jagged arrays (shouldn't happen with current loading)
                    else:
                        var_value = rjr_ms  # For scalar values (expected case)
            
            elif variable_name == 'rjr_Rs_0':
                if file_data['rjr_Rs'] is not None:
                    rjr_rs = file_data['rjr_Rs'][i]
                    # rjr_Rs is already processed to extract [0] element during data loading
                    if hasattr(rjr_rs, '__len__') and len(rjr_rs) > 0:
                        var_value = rjr_rs[0]  # For jagged arrays (shouldn't happen with current loading)
                    else:
                        var_value = rjr_rs  # For scalar values (expected case)
            
            elif variable_name == 'selCMet':
                if file_data['selCMet'] is not None:
                    sel_met = file_data['selCMet'][i]
                    # Handle both array and scalar cases
                    if hasattr(sel_met, '__len__'):
                        if len(sel_met) > 0:
                            var_value = sel_met[0]
                    else:
                        var_value = sel_met
            
            # Add to lists if valid
            if var_value is not None and np.isfinite(var_value):
                values.append(var_value)
                weights.append(base_event_weight)
        
        
        return np.array(values), np.array(weights)

class DataMCComparisonPlotter:
    """Creates Data/MC comparison plots with proper styling."""
    
    def __init__(self, plotter: SVVariablesPlotter):
        self.plotter = plotter
    
    def create_histogram(self, values: np.ndarray, weights: np.ndarray, 
                        name: str, title: str, color: int, bins: np.ndarray) -> ROOT.TH1D:
        """Create and fill histogram with proper styling."""
        n_bins = len(bins) - 1
        hist = ROOT.TH1D(name, "", n_bins, bins)
        
        # Fill histogram
        for val, weight in zip(values, weights):
            hist.Fill(val, weight)
        
        # Set style
        hist.SetLineColor(color)
        hist.SetMarkerColor(color)
        if "data" in name.lower():
            hist.SetMarkerStyle(20)
            hist.SetMarkerSize(0.8)
            hist.SetLineWidth(1)
        else:
            hist.SetLineWidth(2)
            hist.SetFillColor(color)
            hist.SetLineColor(ROOT.kBlack)
            hist.SetLineWidth(1)
        
        return hist
    
    def create_ratio_histogram(self, data_hist: ROOT.TH1D, mc_hist: ROOT.TH1D) -> ROOT.TH1D:
        """Create ratio histogram (data/MC)."""
        ratio_hist = data_hist.Clone("ratio")
        ratio_hist.Divide(mc_hist)
        
        # Style ratio plot
        ratio_hist.SetLineColor(self.plotter.ratio_color)
        ratio_hist.SetMarkerColor(self.plotter.ratio_color)
        ratio_hist.SetMarkerStyle(20)
        ratio_hist.SetMarkerSize(0.8)
        
        return ratio_hist
    
    def create_comparison_canvas(self, data_hist: ROOT.TH1D, mc_histograms: List[Tuple], 
                               total_mc_hist: ROOT.TH1D, variable_name: str, 
                               x_label: str, log_scale: bool, region_label: str = "", normalize: bool = False) -> ROOT.TCanvas:
        """Create two-pad canvas with distribution and ratio plots."""
        canvas = ROOT.TCanvas(f"canvas_{variable_name}", f"{variable_name} Data/MC Comparison", 800, 800)
        
        # Keep references to prevent garbage collection
        canvas.data_hist = data_hist
        canvas.mc_histograms = mc_histograms
        canvas.total_mc_hist = total_mc_hist
        
        # Create pads
        pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.3)
        
        canvas.pad1 = pad1
        canvas.pad2 = pad2
        
        pad1.SetBottomMargin(0.02)
        pad1.SetLeftMargin(0.15)  # 15% left margin for y-axis labels
        pad2.SetTopMargin(0.02)
        pad2.SetLeftMargin(0.15)  # Match pad1's left margin
        pad2.SetBottomMargin(0.4)
        
        canvas.cd()
        pad1.Draw()
        pad2.Draw()
        
        # Draw top pad with stack
        pad1.cd()
        pad1.SetGridx()
        pad1.SetGridy()
        
        # Create THStack
        stack = ROOT.THStack("stack", "")
        canvas.stack = stack
        
        # Add MC histograms to stack
        for mc_hist, _ in mc_histograms:
            stack.Add(mc_hist)
        
        # Only use log scale if all histograms have positive values
        use_log_scale = log_scale
        if log_scale:
            # Check each histogram for negative values by examining bins directly
            has_negative = False
            
            # Check MC histograms
            for mc_hist, _ in mc_histograms:
                for bin_i in range(1, mc_hist.GetNbinsX() + 1):
                    if mc_hist.GetBinContent(bin_i) < 0:
                        has_negative = True
                        break
                if has_negative:
                    break
            
            # Check data histogram if it exists
            if not has_negative and data_hist:
                for bin_i in range(1, data_hist.GetNbinsX() + 1):
                    if data_hist.GetBinContent(bin_i) < 0:
                        has_negative = True
                        break
            
            if has_negative:
                use_log_scale = False
            else:
                pad1.SetLogy()
        
        # Set axis ranges
        data_max = data_hist.GetMaximum() if data_hist else 0
        max_val = max(data_max, stack.GetMaximum()) * 10.
        if use_log_scale:
            stack.SetMinimum(0.0001)  # Set minimum to 0.5 events for log scale
        else:
            stack.SetMinimum(0)
        stack.SetMaximum(max_val)
        
        # Set axis text sizes
        axis_text_size = 0.04
        
        # Draw stack and data
        stack.Draw("HIST")
        stack.GetXaxis().SetLabelSize(0)
        y_axis_title = "normalized events" if normalize else "number of events"
        stack.GetYaxis().SetTitle(y_axis_title)
        stack.GetYaxis().SetTitleSize(axis_text_size+0.01)
        stack.GetYaxis().SetLabelSize(axis_text_size)
        stack.GetYaxis().CenterTitle()
        if data_hist:  # Only draw data if provided (for blinding)
            data_hist.Draw("E SAME")
        
        # Create legend
        n_entries = len(mc_histograms) + (1 if data_hist else 0)
        legend_height = min(0.15, 0.03 * n_entries)
        legend = ROOT.TLegend(0.67, 0.54, 0.97, 0.86)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)  # Make legend transparent
        legend.SetTextSize(0.04)
        
        if data_hist:
            legend.AddEntry(data_hist, "Data", "pe")
        
        # Add MC backgrounds to legend (reverse order for stacking)
        for mc_hist, bg_name in reversed(mc_histograms):
            legend.AddEntry(mc_hist, bg_name, "f")
        
        legend.Draw()
        canvas.legend = legend
        
        # Add CMS logo and luminosity
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextAlign(11)
        latex.SetTextSize(0.065)
        latex.SetTextFont(42)
        
        # CMS preliminary
        latex.SetTextFont(61)
        latex.DrawLatex(0.15, 0.92, "CMS")
        latex.SetTextFont(52)
        latex.SetTextSize(0.05)
        latex.DrawLatex(0.233, 0.92, "Preliminary")
        
        # Luminosity
        latex.SetTextFont(42)
        latex.SetTextAlign(31)
        latex.SetTextSize(0.05)  # Match axis text size
        latex.DrawLatex(0.89, 0.92, f"{self.plotter.luminosity:.0f} fb^{{-1}} (13 TeV)")  # Align with CMS preliminary
        
        # Region label
        if region_label:
            latex.SetTextAlign(11)
            latex.SetTextSize(0.04)
            latex.DrawLatex(0.18, 0.85, region_label)
        
        canvas.latex = latex
        
        # Draw bottom pad (only if data is available)
        if data_hist:
            pad2.cd()
            pad2.SetGridx()
            pad2.SetGridy()
            
            ratio = self.create_ratio_histogram(data_hist, total_mc_hist)
            canvas.ratio = ratio
            
            ratio.GetXaxis().SetTitle(x_label)
            ratio.GetYaxis().SetTitle("#frac{data}{model}")
            ratio.GetYaxis().SetRangeUser(0.5, 1.5)
            ratio.GetXaxis().SetTitleSize(0.12)
            ratio.GetYaxis().SetTitleSize(0.12)
            ratio.GetXaxis().SetLabelSize(0.1)
            ratio.GetYaxis().SetLabelSize(0.09)
            ratio.GetYaxis().SetTitleOffset(0.5)
            ratio.GetXaxis().SetTitleOffset(1.15)
            ratio.GetYaxis().SetNdivisions(505)
            ratio.GetXaxis().CenterTitle()
            ratio.GetYaxis().CenterTitle()
            ratio.Draw("E")
            
            # Reference line at 1
            x_min = ratio.GetXaxis().GetXmin()
            x_max = ratio.GetXaxis().GetXmax()
            line = ROOT.TLine(x_min, 1, x_max, 1)
            line.SetLineStyle(2)
            line.Draw()
            canvas.line = line
        
        canvas.Update()
        return canvas

def create_data_mc_comparison_plots(data_files_data: List[Dict], mc_files_data: List[Dict], 
                                  mc_file_names: List[str], mc_file_scaling: List[float],
                                  region_config: Dict, multiplicity_config: Dict, sv_type: str = 'HadronSV',
                                  variables_to_plot: Optional[List[str]] = None,
                                  blind_data: bool = False, luminosity: float = 400.0,
                                  extra_cuts: Optional[str] = None, normalize: bool = False) -> Dict[str, ROOT.TCanvas]:
    """
    Main function to create all SV variable plots for a specific region.
    
    Args:
        data_files_data: List of loaded data file dictionaries
        mc_files_data: List of loaded MC file dictionaries
        mc_file_scaling: List of scaling factors for each MC file  
        mc_file_names: List of MC file names for labeling
        region_config: Region configuration dict
        multiplicity_config: Multiplicity configuration dict
        sv_type: SV type ('HadronSV', 'LeptonSV', or 'AnySV')
        variables_to_plot: Optional list of specific variables to plot
        blind_data: Whether to blind data (signal regions)
        luminosity: Integrated luminosity in fb-1
        normalize: Whether to normalize data and MC to their total integrals for shape comparison
        
    Returns:
        Dictionary of {variable_name: ROOT.TCanvas} for all plots
    """
    
    # Initialize components
    plotter = SVVariablesPlotter(luminosity)
    selector = RegionSelector()
    comparison_plotter = DataMCComparisonPlotter(plotter)
    
    # Get the appropriate variables for this SV type
    sv_variables = plotter.get_variables_for_sv_type(sv_type)
    
    # Use all variables if none specified
    if variables_to_plot is None:
        variables_to_plot = list(sv_variables.keys())
    
    
    canvases = {}
    
    # Create region label for plots
    region_label = region_config.get('label', '')
    if multiplicity_config.get('label'):
        region_label = f"{multiplicity_config['label']}, {region_label}"
    
    # Process each variable
    for var_name in variables_to_plot:
        if var_name not in sv_variables:
            print(f"Warning: Unknown variable {var_name} for SV type {sv_type}, skipping...")
            continue
        
        bins, x_label, log_scale = sv_variables[var_name]
        
        # Collect data values (if not blinded)
        all_data_values = []
        all_data_weights = []
        
        if not blind_data:
            for file_data in data_files_data:
                # Apply region selection
                event_mask = selector.apply_region_selection(file_data, region_config, multiplicity_config, sv_type, extra_cuts, 'data')
                
                # Extract variable values
                values, weights = selector.extract_variable_values(file_data, 'data', var_name, event_mask, luminosity)
                all_data_values.extend(values)
                all_data_weights.extend(weights)
        
        # Process MC files
        mc_histograms = []
        
        for i, (file_data, mc_file) in enumerate(zip(mc_files_data, mc_file_names)):
            # Apply region selection
            event_mask = selector.apply_region_selection(file_data, region_config, multiplicity_config, sv_type, extra_cuts, 'mc')
            
            # Extract variable values
            values, weights = selector.extract_variable_values(file_data, 'mc', var_name, event_mask, luminosity)
            
            # Apply scaling factor
            scaling_factor = mc_file_scaling[i]
            if scaling_factor != 1.0:
                weights = weights * scaling_factor
            
            if len(values) > 0:
                # Extract background name from filename
                raw_bg_name = os.path.basename(mc_file).split('_background')[0] if '_background' in mc_file else os.path.basename(mc_file).replace('.root', '')
                bg_name = plotter._clean_mc_label(raw_bg_name)
                color = plotter.mc_colors[i % len(plotter.mc_colors)]  # Color assigned by file order
                
                # Create unique histogram name including region and multiplicity
                unique_name = f"mc_{bg_name}_{var_name}_{region_config.get('name', 'unknown')}_{multiplicity_config.get('name', 'unknown')}"
                mc_hist = comparison_plotter.create_histogram(values, weights, 
                                                            unique_name, bg_name, color, bins)
                # Store histogram with yield for sorting and original file index for color consistency
                mc_histograms.append((mc_hist, bg_name, mc_hist.Integral(), i))
        
        # Sort MC histograms by yield (ascending order - smallest at bottom of stack)
        mc_histograms.sort(key=lambda x: x[2])  # Sort by integral (3rd element)
        
        # Convert back to (hist, name) tuples for compatibility with existing code
        mc_histograms = [(hist, name) for hist, name, integral, file_idx in mc_histograms]
        
        # Create data histogram (if not blinded)
        data_hist = None
        if not blind_data and len(all_data_values) > 0:
            data_values = np.array(all_data_values)
            data_weights = np.array(all_data_weights)
            # Create unique data histogram name
            unique_data_name = f"data_{var_name}_{region_config.get('name', 'unknown')}_{multiplicity_config.get('name', 'unknown')}"
            data_hist = comparison_plotter.create_histogram(data_values, data_weights, 
                                                          unique_data_name, "Data", plotter.data_color, bins)
        
        # Create total MC histogram for ratio
        total_mc_hist = None
        if mc_histograms:
            # Create unique total MC histogram name
            unique_total_name = f"total_mc_{var_name}_{region_config.get('name', 'unknown')}_{multiplicity_config.get('name', 'unknown')}"
            total_mc_hist = mc_histograms[0][0].Clone(unique_total_name)
            total_mc_hist.Reset()
            for mc_hist, _ in mc_histograms:
                total_mc_hist.Add(mc_hist)
        
        # Apply normalization if requested
        if normalize and mc_histograms:
            # Get total MC integral (after scaling)
            total_mc_integral = total_mc_hist.Integral() if total_mc_hist else 0
            
            # Get data integral
            data_integral = data_hist.Integral() if data_hist else 0
            
            if total_mc_integral > 0:
                # Normalize MC histograms to unit area
                mc_norm_factor = 1.0 / total_mc_integral
                for mc_hist, _ in mc_histograms:
                    mc_hist.Scale(mc_norm_factor)
                total_mc_hist.Scale(mc_norm_factor)
                
                # Normalize data histogram to unit area
                if data_hist and data_integral > 0:
                    data_norm_factor = 1.0 / data_integral
                    data_hist.Scale(data_norm_factor)
        
        # Create comparison canvas
        if mc_histograms:
            suffix = "_mc_only" if blind_data else "_comparison"
            # Create unique canvas name
            unique_canvas_name = f"{var_name}_{region_config.get('name', 'unknown')}_{multiplicity_config.get('name', 'unknown')}{suffix}"
            try:
                canvas = comparison_plotter.create_comparison_canvas(
                    data_hist, mc_histograms, total_mc_hist, 
                    unique_canvas_name, x_label, log_scale, region_label, normalize
                )
                if canvas is not None:
                    canvases[var_name] = canvas
            except Exception as e:
                print(f"Warning: Failed to create canvas for {var_name}: {e}")
    
    return canvases

def save_plots_to_root_directory(canvases: Dict[str, ROOT.TCanvas], root_directory: ROOT.TDirectory):
    """Save canvases to a ROOT directory."""
    root_directory.cd()
    for var_name, canvas in canvases.items():
        if canvas is not None:
            canvas.Write()
        else:
            print(f"Warning: Canvas for {var_name} is None, skipping ROOT save")

def save_plots_to_pdf_folder(canvases: Dict[str, ROOT.TCanvas], pdf_folder_path: str):
    """Save canvases as individual PDF files in a folder."""
    os.makedirs(pdf_folder_path, exist_ok=True)
    
    for var_name, canvas in canvases.items():
        if canvas is not None:
            pdf_path = os.path.join(pdf_folder_path, f"{var_name}.pdf")
            canvas.SaveAs(pdf_path)
        else:
            print(f"Warning: Canvas for {var_name} is None, skipping PDF save")
