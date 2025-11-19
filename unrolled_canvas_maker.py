#!/usr/bin/env python3
"""
UnrolledCanvasMaker - Canvas creation and plot finalization for unrolled plots.

This class handles:
- Creating properly configured canvases 
- Adding separator lines between groups
- Adding group labels and axis titles
- Adding CMS labels and luminosity info
- Managing plot layout and final presentation

Designed to work with UnrolledDataProcessor and UnrolledHistogramMaker.
"""

import ROOT
import numpy as np
from typing import Dict, List, Tuple, Optional
from plotting import Plot

class UnrolledCanvasMaker:
    def __init__(self, luminosity: float = 400.0):
        """
        Initialize the canvas maker.
        
        Args:
            luminosity: Integrated luminosity in fb-1
        """
        self.luminosity = luminosity
        
        # Pre-create custom colors at initialization
        self._register_custom_colors()
        
        # Canvas configuration
        self.canvas_config = {
            'width': 1200,
            'height': 600,
            'bottom_margin': 0.15,
            'left_margin': 0.12,
            'right_margin': 0.15
        }
        
        # Group label configuration
        self.label_config = {
            'group_label_size': 0.038,
            'group_label_y_position': 0.87,  # Near top of canvas
            'separator_line_width': 2,
            'separator_line_color': ROOT.kBlack
        }
    
    def _register_custom_colors(self):
        """Register custom colors and store their indices."""
        hex_colors = ["#5A4484", "#347889", "#F4B240", "#E54B26", "#C05780", "#7A68A6", "#2E8B57", "#8B4513"]
        self.custom_colors = []
        
        print(f"Registering {len(hex_colors)} custom colors...")
        for i, hex_color in enumerate(hex_colors):
            color_index = ROOT.TColor.GetColor(hex_color)
            self.custom_colors.append(color_index)
            print(f"  Color {i}: {hex_color} -> index {color_index}")
        
        print(f"Custom colors registered: {self.custom_colors}")
    
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
    
    def _format_sv_label(self, final_state: str) -> str:
        """
        Format final state labels according to SV convention: {N}SV_{flavor}^{selection}
        
        Convention:
        - N: total count of SVs (only shown for 2 or more, just shows "2")
        - flavor: 'hh' (hadronic) or '\\ell\\ell' (leptonic)
        - selection: CRL, SRL, CRT, SRT
        - mixed case: SV_{\\ell\\ell}SV_{hh}^{selection}
        
        Examples:
        - passNHad1SelectionSRLoose -> SV_{hh}^{SRL} (single hadronic, signal region loose)
        - passNLep1SelectionSRTight -> SV_{\\ell\\ell}^{SRT} (single leptonic, signal region tight)
        - passNHad2SelectionCRLoose -> 2SV_{hh}^{CRL} (two hadronic, control region loose)
        - passNLepNHadSelectionSRTight -> SV_{\\ell\\ell}SV_{hh}^{SRT} (mixed)
        """
        import re
        
        # Default values
        count = ""
        flavor = "hh"  # default to hadronic
        selection = "SRT"  # default to signal region tight
        
        # Parse the branch name pattern: passN{Had|Lep}{count}Selection{SR|CR}{Loose|Tight}
        
        # Extract flavor and count
        if "HadAndLep" in final_state or "LepAndHad" in final_state:
            # Mixed case - still need to extract selection
            if "CR" in final_state:
                if "Loose" in final_state:
                    selection = "CR,L"
                else:
                    selection = "CR,T"
            elif "SR" in final_state:
                if "Loose" in final_state:
                    selection = "SR,L"
                else:
                    selection = "SR,T"
            return f"SV_{{\\ell\\ell}}SV_{{hh}}^{{{selection}}}"
        elif "NHad" in final_state:
            flavor = "hh"  # hadronic
            # Check for Ge2 (2 or more) pattern
            if "HadGe2" in final_state:
                count = "2"
            else:
                # Extract count after NHad
                had_match = re.search(r'NHad(\d+)', final_state)
                if had_match:
                    sv_count = int(had_match.group(1))
                    if sv_count >= 2:
                        count = "2"
        elif "NLep" in final_state:
            flavor = "\\ell\\ell"  # leptonic
            # Check for Ge2 (2 or more) pattern  
            if "LepGe2" in final_state:
                count = "2"
            else:
                # Extract count after NLep
                lep_match = re.search(r'NLep(\d+)', final_state)
                if lep_match:
                    sv_count = int(lep_match.group(1))
                    if sv_count >= 2:
                        count = "2"
        
        # Extract selection region
        if "CR" in final_state:
            if "Loose" in final_state:
                selection = "CR,L"
            else:
                selection = "CR,T"
        elif "SR" in final_state:
            if "Loose" in final_state:
                selection = "SR,L"
            else:
                selection = "SR,T"
        
        # Format final label
        return f"{count}SV_{{{flavor}}}^{{{selection}}}"

    #def _get_final_state_label(self, label: str) -> str:

        
    
    def create_base_canvas(self, name: str, title: str = "", 
                          use_log_y: bool = True, use_grid: bool = True) -> ROOT.TCanvas:
        """
        Create a base canvas with proper configuration.
        
        Args:
            name: Canvas name
            title: Canvas title (usually empty)
            use_log_y: Enable logarithmic y-axis
            use_grid: Enable grid lines
            
        Returns:
            Configured ROOT.TCanvas
        """
        # Create canvas
        canvas = ROOT.TCanvas(name, title, 
                            self.canvas_config['width'], 
                            self.canvas_config['height'])
        
        # Set margins (bottom, left, and right)
        canvas.SetBottomMargin(self.canvas_config['bottom_margin'])
        canvas.SetLeftMargin(self.canvas_config['left_margin'])
        canvas.SetRightMargin(self.canvas_config['right_margin'])
        
        # Set log scale and grid
        if use_log_y:
            canvas.SetLogy()
        if use_grid:
            canvas.SetGridx()
            canvas.SetGridy()
        
        return canvas
    
    def add_histogram_to_canvas(self, canvas: ROOT.TCanvas, hist: ROOT.TH1D, 
                               draw_option: str = "hist", is_first: bool = True) -> None:
        """
        Add a histogram to the canvas with proper scaling.
        
        Args:
            canvas: Canvas to draw on
            hist: Histogram to add
            draw_option: ROOT draw option
            is_first: Whether this is the first histogram (sets axes)
        """
        canvas.cd()
        
        if is_first:
            # First histogram sets up axes
            hist.Draw(draw_option)
        else:
            # Subsequent histograms use "same"
            if "same" not in draw_option.lower():
                draw_option += " same"
            hist.Draw(draw_option)
        
        canvas.Update()
    
    def add_error_band_to_canvas(self, canvas: ROOT.TCanvas, 
                                error_graph: ROOT.TGraphAsymmErrors) -> None:
        """
        Add an error band to the canvas.
        
        Args:
            canvas: Canvas to draw on
            error_graph: Error graph to add
        """
        if error_graph is not None:
            canvas.cd()
            error_graph.Draw("2 same")  # Filled error band
            canvas.Update()
    

    def add_separator_lines(self, canvas: ROOT.TCanvas, hist: ROOT.TH1D):

        """
         Add separator lines between groups.
         Args:
            canvas: Canvas to add lines to
            hist: Histogram for range information
        Returns:
            List of line objects (for memory management)
         """
                                  
        canvas.cd()
        canvas.Update()
        
        # === Create an overlay pad (fully visual coordinate system, NDC) ===
        overlay = ROOT.TPad("overlay","overlay",0,0,1,1)
        overlay.SetFillStyle(0)
        overlay.SetFrameFillStyle(0)
        overlay.SetBorderSize(0)
        overlay.SetBorderMode(0)
        overlay.SetMargin(0,0,0,0)
        overlay.SetBit(ROOT.kCannotPick)  # Make transparent to mouse events
        overlay.Draw()
        overlay.cd()
        
        # === Convert x positions to NDC within the *main* pad ===
        main = canvas.GetPad(0)
        main.Update()
        
        x_axis = hist.GetXaxis()
        x_min = x_axis.GetXmin()
        x_max = x_axis.GetXmax()
        
        # Pad margins
        left_ndc = main.GetLeftMargin()
        right_ndc = 1.0 - main.GetRightMargin()
        data_ndc_width = right_ndc - left_ndc
        
        is_logx = bool(main.GetLogx())
        import math

        def normalized_x(xval):
            if is_logx:
                return (math.log(xval) - math.log(x_min)) / (math.log(x_max)-math.log(x_min))
            else:
                return (xval - x_min) / (x_max - x_min)

        def x_to_ndc_from_bin(bin_index):
            x_edge = x_axis.GetBinLowEdge(bin_index+1)
            norm = normalized_x(x_edge)
            return left_ndc + norm * data_ndc_width

        # === Now draw the vertical lines in PURE NDC ===
        y_bottom = 0.07   # constant, visual extent ONLY
        y_top    = 0.9
        
        lines = []
        for b in [3,6]:
            x_ndc = x_to_ndc_from_bin(b)
            line = ROOT.TLine()
            line.SetNDC(True)
            line.SetLineColor(self.label_config['separator_line_color'])
            line.SetLineWidth(self.label_config['separator_line_width'])
            line.SetLineStyle(1)
            line.DrawLine(x_ndc, y_bottom, x_ndc, y_top)
            lines.append(line)

        canvas.Modified()
        canvas.Update()
        return lines

    
    def add_group_labels(self, canvas: ROOT.TCanvas, group_labels: List[str]) -> List[ROOT.TLatex]:
        """
        Add group labels at the top of the plot.
        
        Args:
            canvas: Canvas to add labels to
            group_labels: List of group label strings
            
        Returns:
            List of TLatex objects (for memory management)
        """
        # Draw on overlay pad 
        overlay_pad = canvas.GetListOfPrimitives().FindObject("overlay")
        overlay_pad.cd()
        
        # Get actual pad boundaries in NDC from main pad
        main_pad = canvas.GetPad(0)
        pad_left = main_pad.GetLeftMargin()
        pad_right = 1.0 - main_pad.GetRightMargin()
        pad_width = pad_right - pad_left
        
        # Each group covers 3 bins out of 9 total bins
        group_width_ndc = pad_width / 3
        
        text_objects = []
        latex = ROOT.TLatex()
        latex.SetTextAlign(22)  # Center alignment
        latex.SetTextSize(self.label_config['group_label_size'])
        latex.SetTextFont(42)  # Helvetica (normal, not bold)
        latex.SetNDC(True)  # Use NDC coordinates
        
        for i, group_name in enumerate(group_labels):
            # Calculate center of each group based on actual pad margins
            group_center_ndc = pad_left + group_width_ndc * (i + 0.5)
            y_ndc = self.label_config['group_label_y_position']
            
            text_obj = latex.DrawLatex(group_center_ndc, y_ndc, group_name)
            text_objects.append(text_obj)
        
        return text_objects
    
    def universal_cms_mark(self, cms_x: float, cms_y: float, text_size: float, 
                           preliminary_x: float = None, preliminary_y: float = None,
                           cms_text: str = "CMS", preliminary_text: str = "Preliminary", ) -> List[ROOT.TLatex]:
        """
        Universal CMS mark function with configurable positions and sizes.
        
        Args:
            cms_x: X position for CMS text (NDC coordinates)
            cms_y: Y position for CMS text (NDC coordinates)  
            text_size: Base text size (CMS text will be 1.3x larger)
            preliminary_x: X position for preliminary text (defaults to cms_x + 0.06)
            preliminary_y: Y position for preliminary text (defaults to cms_y)
            cms_text: CMS text (default: "CMS")
            preliminary_text: Preliminary text (default: "Preliminary")
            
        Returns:
            List of TLatex objects for memory management
        """
        if preliminary_x is None:
            preliminary_x = cms_x + 0.06
        if preliminary_y is None:
            preliminary_y = cms_y
            
        latex_objects = []
        
        # Draw CMS text (1.3x larger than base text size)
        cms_latex = ROOT.TLatex()
        cms_latex.SetNDC()
        cms_latex.SetTextAlign(11)  # Left bottom align
        cms_latex.SetTextFont(61)   # Bold font
        cms_latex.SetTextSize(text_size * 1.3)
        cms_latex.DrawLatex(cms_x, cms_y, cms_text)
        latex_objects.append(cms_latex)
        
        # Draw preliminary text (base text size)
        prelim_latex = ROOT.TLatex()
        prelim_latex.SetNDC() 
        prelim_latex.SetTextAlign(11)  # Left bottom align
        prelim_latex.SetTextFont(52)   # Italic font
        prelim_latex.SetTextSize(text_size)
        prelim_latex.DrawLatex(preliminary_x, preliminary_y, preliminary_text)
        latex_objects.append(prelim_latex)
        
        return latex_objects

    def add_cms_labels(self, canvas: ROOT.TCanvas,
                       x_location: float = 0.12,
                       y_location: float = 0.915,
                       text_size: float = 0.04) -> List[ROOT.TLatex]:
        """
        Add CMS preliminary mark and luminosity label.
        
        Args:
            canvas: Canvas to add labels to
            
        Returns:
            List of TLatex objects (for memory management)
        """
        # Draw on overlay pad
        overlay_pad = canvas.GetListOfPrimitives().FindObject("overlay")
        overlay_pad.cd()
        
        # Use universal CMS mark
        cms_objects = self.universal_cms_mark(x_location, y_location, text_size, preliminary_x=x_location+0.056)
        
        # Add luminosity label
        lumi_latex = ROOT.TLatex()
        lumi_latex.SetTextFont(42)
        lumi_latex.SetNDC()
        lumi_latex.SetTextSize(text_size)
        lumi_latex.SetTextAlign(31)  # Right align
        lumi_latex.DrawLatex(0.85, y_location, f"{self.luminosity:.0f} fb^{{-1}} (13 TeV)")
        
        return cms_objects + [lumi_latex]
    
    def finalize_canvas(self, canvas: ROOT.TCanvas, hist: ROOT.TH1D, 
                       group_labels: List[str], error_band: Optional[ROOT.TGraphAsymmErrors] = None,
                       additional_hists: List[ROOT.TH1D] = None) -> None:
        """
        Finalize canvas with all decorations.
        
        Args:
            canvas: Canvas to finalize
            hist: Main histogram
            group_labels: Group labels to add
            error_band: Optional error band
            additional_hists: Additional histograms for overlays
        """
        # Draw main histogram
        self.add_histogram_to_canvas(canvas, hist, "hist", is_first=True)
        
        # Add error band if provided
        if error_band is not None:
            self.add_error_band_to_canvas(canvas, error_band)
        
        # Add additional histograms if provided
        if additional_hists:
            for add_hist in additional_hists:
                self.add_histogram_to_canvas(canvas, add_hist, "hist", is_first=False)
        
        # Add separator lines 
        separator_lines = self.add_separator_lines(canvas, hist)
        
        # Add group labels 
        text_objects = self.add_group_labels(canvas, group_labels)
        
        canvas.Modified()
        canvas.Update()
        
        # Draw CMS marks and luminosity on overlay pad
        overlay_pad = canvas.GetListOfPrimitives().FindObject("overlay")
        overlay_pad.cd()
        
        # Use universal CMS mark
        cms_objects = self.universal_cms_mark(0.12, 0.91, 0.04)

        # Add luminosity label exactly like original
        lumi_latex = ROOT.TLatex()
        lumi_latex.SetTextFont(42)
        lumi_latex.SetNDC()
        lumi_latex.SetTextSize(0.04)
        lumi_latex.SetTextAlign(31)
        lumi_latex.DrawLatex(0.72, 0.91, f"{self.luminosity:.0f} fb^{{-1}} (13 TeV)")
        canvas.lumi_latex = lumi_latex
        canvas.cms_objects_finalize = cms_objects
        
        # Store objects to prevent garbage collection
        canvas.lines = separator_lines
        canvas.text_objects = text_objects
        canvas.histogram = hist
        if error_band is not None:
            canvas.error_graph = error_band
        if additional_hists:
            canvas.additional_hists = additional_hists
    
    def save_canvas(self, canvas: ROOT.TCanvas, output_path: str, 
                   formats: List[str] = ['pdf']) -> None:
        """
        Save canvas in specified formats.
        
        Args:
            canvas: Canvas to save
            output_path: Base output path (without extension)
            formats: List of formats ('pdf', 'png', 'root', 'eps', 'svg')
        """
        import os
        
        # Check if we have any non-root formats
        has_non_root = any(fmt != 'root' for fmt in formats)
        
        for fmt in formats:
            if fmt == 'root':
                # Save as ROOT file with canvas
                root_file = ROOT.TFile(f"{output_path}.root", "RECREATE")
                canvas.Write()
                root_file.Close()
            else:
                if has_non_root:
                    # Create folder for non-root formats
                    base_name = os.path.basename(output_path)
                    dir_name = os.path.dirname(output_path)
                    
                    # Create output folder named after the output file
                    output_folder = os.path.join(dir_name, base_name) if dir_name else base_name
                    os.makedirs(output_folder, exist_ok=True)
                    
                    # Save inside the folder with descriptive name
                    filename = f"{canvas.GetName()}.{fmt}"
                    file_path = os.path.join(output_folder, filename)
                else:
                    file_path = f"{output_path}.{fmt}"
                
                if fmt == 'png':
                    self._save_high_res_png(canvas, file_path)
                else:
                    # Save as other image formats
                    canvas.SaveAs(file_path)
                    
    def save_canvases_to_folder(self, canvases: Dict[str, ROOT.TCanvas], 
                               base_output_path: str, formats: List[str] = ['pdf']) -> None:
        """
        Save multiple canvases to a single shared folder for non-root formats.
        
        Args:
            canvases: Dictionary of {name: canvas} pairs
            base_output_path: Base output path (without extension) 
            formats: List of formats ('pdf', 'png', 'root', 'eps', 'svg')
        """
        import os
        
        # Handle root format separately (single file)
        if 'root' in formats:
            self.save_to_root_file(canvases, base_output_path)
            
        # Handle non-root formats (shared folder)
        non_root_formats = [fmt for fmt in formats if fmt != 'root']
        if non_root_formats:
            # Create shared output folder
            base_name = os.path.basename(base_output_path)
            dir_name = os.path.dirname(base_output_path)
            
            output_folder = os.path.join(dir_name, base_name) if dir_name else base_name
            os.makedirs(output_folder, exist_ok=True)
            
            # Save each canvas to the shared folder
            for canvas_name, canvas in canvases.items():
                for fmt in non_root_formats:
                    filename = f"{canvas_name}.{fmt}"
                    file_path = os.path.join(output_folder, filename)
                    
                    if fmt == 'png':
                        self._save_high_res_png(canvas, file_path)
                    else:
                        # Save as other image formats
                        canvas.SaveAs(file_path)
                        
    def save_to_root_file(self, canvases: Dict[str, ROOT.TCanvas], output_path: str) -> None:
        """Save multiple canvases to a single ROOT file."""
        root_file = ROOT.TFile(f"{output_path}.root", "RECREATE")
        for canvas_name, canvas in canvases.items():
            canvas.Write()
        root_file.Close()
        
    def _save_high_res_png(self, canvas: ROOT.TCanvas, file_path: str) -> None:
        """Save canvas as high-resolution PNG preserving original aspect ratio."""
        # Get original canvas dimensions 
        orig_w = canvas.GetWw()
        orig_h = canvas.GetWh()
        
        # Scale up by 2x for higher resolution while preserving aspect ratio
        scale_factor = 2.0
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        
        # Create a temporary high-resolution canvas with same aspect ratio
        temp_canvas = ROOT.TCanvas(f"temp_{canvas.GetName()}", "", new_w, new_h)
        temp_canvas.SetMargin(canvas.GetLeftMargin(), canvas.GetRightMargin(), 
                            canvas.GetBottomMargin(), canvas.GetTopMargin())
        temp_canvas.SetLogx(canvas.GetLogx())
        temp_canvas.SetLogy(canvas.GetLogy())
        temp_canvas.SetLogz(canvas.GetLogz())
        temp_canvas.SetGridx(canvas.GetGridx())
        temp_canvas.SetGridy(canvas.GetGridy())
        temp_canvas.SetTickx(canvas.GetTickx())
        temp_canvas.SetTicky(canvas.GetTicky())
        
        # Copy all primitives from original canvas
        canvas.cd()
        temp_canvas.cd()
        canvas.DrawClonePad()
        
        # Make frame and tick marks thicker for PNG
        temp_canvas.cd()
        frame = temp_canvas.GetFrame()
        if frame:
            frame.SetLineWidth(3)  # Thicker frame border
        
        # Make tick marks thicker by adjusting axis properties
        primitives = temp_canvas.GetListOfPrimitives()
        for primitive in primitives:
            # Check if it's a histogram or graph with axes
            if hasattr(primitive, 'GetXaxis'):
                primitive.GetXaxis().SetTickLength(0.02)  # Slightly longer ticks
                primitive.GetXaxis().SetLineWidth(3)      # Thicker axis lines
            if hasattr(primitive, 'GetYaxis'):
                primitive.GetYaxis().SetTickLength(0.02)  # Slightly longer ticks  
                primitive.GetYaxis().SetLineWidth(3)      # Thicker axis lines
        
        temp_canvas.Modified()
        temp_canvas.Update()
        
        # Save the high-res version
        temp_canvas.SaveAs(file_path)
        temp_canvas.Close()
        del temp_canvas


                
    def create_offset_graph(self, hist: ROOT.TH1D, offset_fraction: float, 
                           marker_style: int = 20, marker_size: float = 1.0, 
                           marker_color: int = ROOT.kBlack) -> ROOT.TGraphErrors:
        """
        Convert histogram to TGraph with x-offset markers for jittered plotting.
        
        Args:
            hist: Input histogram
            offset_fraction: Fraction of bin width to offset (between -0.5 and 0.5)
            marker_style: ROOT marker style
            marker_size: Marker size
            marker_color: Marker color
            
        Returns:
            TGraphErrors with offset x positions
        """
        n_bins = hist.GetNbinsX()
        x_vals = []
        y_vals = []
        x_errors = []
        y_errors = []
        
        for i in range(1, n_bins + 1):
            bin_center = hist.GetBinCenter(i)
            bin_content = hist.GetBinContent(i)
            bin_error = hist.GetBinError(i)
            bin_width = hist.GetBinWidth(i)
            
            # Apply offset as fraction of bin width
            x_offset = offset_fraction * bin_width
            x_position = bin_center + x_offset
            
            x_vals.append(x_position)
            y_vals.append(bin_content)
            x_errors.append(0.0)  # No x error bars
            y_errors.append(bin_error)
        
        # Create TGraphErrors
        graph = ROOT.TGraphErrors(len(x_vals), 
                                np.array(x_vals, dtype=float),
                                np.array(y_vals, dtype=float),
                                np.array(x_errors, dtype=float), 
                                np.array(y_errors, dtype=float))
        
        # Style the graph
        graph.SetMarkerStyle(marker_style)
        graph.SetMarkerSize(marker_size)
        graph.SetMarkerColor(marker_color)
        graph.SetLineColor(marker_color)
        graph.SetLineWidth(2)
        
        return graph

    def create_legend(self, entries: List[Dict], x1: float = 0.77, y1: float = 0.8,
                      x2: float = 1., y2: float = 0.88) -> ROOT.TLegend:
        """
        Create a legend for multiple histograms.
        
        Args:
            entries: List of dicts with 'object', 'label', 'option' keys
            x1, y1, x2, y2: Legend position in NDC coordinates
            
        Returns:
            ROOT.TLegend object
        """
        legend = ROOT.TLegend(x1, y1, x2, y2)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetMargin(0.15)  
        legend.SetEntrySeparation(0.5)
        
        # Use two columns if more than 3 entries
        #if len(entries) > 3:
        #legend.SetNColumns(2)
        #legend.SetColumnSeparation(-0.25)
        
        for entry in entries:
            legend.AddEntry(entry['object'], entry['label'], entry['option'])
        
        return legend
    
    def add_legend_to_canvas(self, canvas: ROOT.TCanvas, legend: ROOT.TLegend) -> None:
        """
        Add legend to canvas.
        
        Args:
            canvas: Canvas to add legend to
            legend: Legend object
        """
        # Draw on main canvas for interactivity
        canvas.cd()
        legend.Draw()
        canvas.legend = legend  # Prevent garbage collection
    
    def create_comparison_canvas(self, histograms: List[ROOT.TH1D], labels: List[str],
                                group_labels: List[str], name: str, 
                                draw_options: List[str] = None) -> ROOT.TCanvas:
        """
        Create a canvas with multiple overlaid histograms.
        
        Args:
            histograms: List of histograms to overlay
            labels: Legend labels for each histogram
            group_labels: Group labels for the plot
            name: Canvas name
            draw_options: Draw options for each histogram
            
        Returns:
            Canvas with overlaid histograms and legend
        """
        if not histograms:
            raise ValueError("No histograms provided")
        
        if draw_options is None:
            draw_options = ["hist"] * len(histograms)
        
        # Create base canvas
        canvas = self.create_base_canvas(f"{name}_comparison")
        
        # Find maximum for proper scaling (only set on first histogram to establish axes)
        max_val = max(hist.GetMaximum() for hist in histograms)
        histograms[0].SetMaximum(max_val * 5.)  # 5x headroom for log scale, only on first histogram
        
        # Draw histograms
        for i, (hist, draw_opt) in enumerate(zip(histograms, draw_options)):
            self.add_histogram_to_canvas(canvas, hist, draw_opt, is_first=(i == 0))
        
        # Create legend first (on main canvas)
        legend_entries = [
            {'object': hist, 'label': label, 'option': 'l'}
            for hist, label in zip(histograms, labels)
        ]
        legend = self.create_legend(legend_entries, 0.77, 0.51, 1.1, 0.88)
        self.add_legend_to_canvas(canvas, legend)
        
        # Add decorations (creates overlay pad)
        separator_lines = self.add_separator_lines(canvas, histograms[0])
        text_objects = self.add_group_labels(canvas, group_labels)
        cms_objects = self.add_cms_labels(canvas)
        
        # Store objects
        canvas.lines = separator_lines
        canvas.text_objects = text_objects
        canvas.cms_objects = cms_objects
        canvas.histograms = histograms
        
        return canvas
    
    def create_comparison_canvas_with_markers(self, histograms: List[ROOT.TH1D], labels: List[str],
                                            group_labels: List[str], name: str, 
                                            marker_styles: List[int] = None, 
                                            original_yields: List[float] = None) -> ROOT.TCanvas:
        """
        Create a canvas with multiple overlaid histograms using offset markers for jittered visualization.
        
        Args:
            histograms: List of histograms to overlay
            labels: Legend labels for each histogram
            group_labels: Group labels for the plot
            name: Canvas name
            marker_styles: Custom marker styles for each histogram
            
        Returns:
            Canvas with overlaid marker graphs and legend
        """
        if not histograms:
            raise ValueError("No histograms provided")
        
        n_hists = len(histograms)
        
        # Default marker styles if not provided
        if marker_styles is None:
            marker_styles = [20, 21, 22, 23, 47, 25, 26, 32, 33]  # Different marker styles
        
        # Create base canvas
        canvas = self.create_base_canvas(f"{name}_comparison_markers")
        
        # Find maximum for proper scaling
        max_val = max(hist.GetMaximum() for hist in histograms)
        
        # Create first histogram as invisible for axis setup
        axis_hist = histograms[0].Clone(f"{name}_axis_template")
        axis_hist.SetMaximum(max_val * 5.)  # 5x headroom for log scale
        axis_hist.SetLineColor(0)  # Invisible
        axis_hist.SetMarkerSize(0)  # No markers
        axis_hist.SetFillStyle(0)  # No fill
        self.add_histogram_to_canvas(canvas, axis_hist, "hist", is_first=True)
        
        # Ensure grid is visible after drawing axis
        canvas.SetGridx()
        canvas.SetGridy()
        canvas.Update()
        
        # Sort histograms by total yield (highest to lowest for display order)
        hist_with_yields = []
        for i, hist in enumerate(histograms):
            # Use original yields if provided, otherwise fall back to histogram integral
            if original_yields and i < len(original_yields):
                yield_total = original_yields[i]
            else:
                yield_total = hist.Integral()
            hist_with_yields.append((yield_total, hist, labels[i], i))
        
        # Sort by yield (highest first)
        hist_with_yields.sort(key=lambda x: x[0], reverse=True)
        
        
        # Convert histograms to offset graphs and draw them
        graphs = []
        legend_entries = []
        
        for display_idx, (yield_total, hist, label, orig_idx) in enumerate(hist_with_yields):
            # Calculate offset: spread evenly across bin width
            # For n histograms, offsets go from -0.4 to +0.4 (leaving 20% margin on each side)
            if n_hists == 1:
                offset = 0.0
            else:
                offset = -0.4 + (0.8 * display_idx / (n_hists - 1))
            
            # Get histogram color for original index
            from unrolled_histogram_maker import UnrolledHistogramMaker
            hist_maker = UnrolledHistogramMaker()
            color = hist_maker.comparison_colors[orig_idx % len(hist_maker.comparison_colors)]
            marker_style = marker_styles[orig_idx % len(marker_styles)]
            
            # Create offset graph
            graph = self.create_offset_graph(hist, offset, marker_style=marker_style, 
                                           marker_size=1.4, marker_color=color)
            
            graphs.append(graph)
            legend_entries.append({'object': graph, 'label': label, 'option': 'p'})
        
        # Draw all graphs
        canvas.cd()
        for graph in graphs:
            graph.Draw("P SAME")  # P = markers, SAME = on same canvas
        
        canvas.Update()
        
        # Create legend
        legend = self.create_legend(legend_entries, 0.85, 0.5, 1.05, 0.91)
        self.add_legend_to_canvas(canvas, legend)
        
        # Add decorations (creates overlay pad)
        separator_lines = self.add_separator_lines(canvas, histograms[0])
        text_objects = self.add_group_labels(canvas, group_labels)
        cms_objects = self.add_cms_labels(canvas)
        
        # Store objects
        canvas.lines = separator_lines
        canvas.text_objects = text_objects
        canvas.cms_objects = cms_objects
        canvas.graphs = graphs
        canvas.axis_hist = axis_hist
        
        return canvas
    
    def _create_base_ratio_canvas(self, name: str, normalize: bool = False) -> Tuple[ROOT.TCanvas, ROOT.TPad, ROOT.TPad]:
        """
        Create base two-pad canvas structure for ratio plots.
        
        Args:
            name: Canvas name
            normalize: Whether this will be used for normalized plots
            
        Returns:
            Tuple of (canvas, top_pad, bottom_pad)
        """
        # Create canvas with increased right margin for external legend
        canvas = ROOT.TCanvas(name, "", 
                            self.canvas_config['width'] + 100, 
                            self.canvas_config['height'] + 100)
        
        # Create two pads: top for distribution, bottom for ratio
        pad1 = ROOT.TPad("pad1", "Distribution", 0, 0.3, 0.8, 1.0)  # 85% width for plot area
        pad2 = ROOT.TPad("pad2", "Ratio", 0, 0.0, 0.8, 0.3)         # 85% width for ratio
        
        # Set margins for main plotting area
        pad1.SetLeftMargin(0.15)
        pad1.SetRightMargin(0.015)
        pad1.SetBottomMargin(0.0)   # Remove gap between pads
        pad1.SetTopMargin(0.08)
        
        pad2.SetLeftMargin(0.15)
        pad2.SetRightMargin(0.015)
        pad2.SetTopMargin(0.0)      # Remove gap between pads
        pad2.SetBottomMargin(0.4)
        
        # Enable log scale and grid
        pad1.SetLogy()
        pad1.SetGridx()
        pad1.SetGridy()
        pad1.SetTickx(1)  # Add tick marks on top
        pad1.SetTicky(1)  # Add tick marks on right side
        
        # Set smaller tick marks on the pad
        ROOT.gStyle.SetTickLength(0.02, "X")
        ROOT.gStyle.SetTickLength(0.02, "Y")
        
        pad2.SetGridx()
        pad2.SetGridy()
        pad2.SetTickx(1)  # Add tick marks on top
        pad2.SetTicky(1)  # Add tick marks on right side
        
        canvas.cd()
        pad1.Draw()
        pad2.Draw()
        
        return canvas, pad1, pad2

    def _create_ratio_histogram(self, numerator: ROOT.TH1D, denominator: ROOT.TH1D, 
                               name: str, ratio_title: str = "#frac{data}{model}") -> ROOT.TH1D:
        """
        Create ratio histogram with proper styling.
        
        Args:
            numerator: Numerator histogram
            denominator: Denominator histogram
            name: Name for ratio histogram
            ratio_title: Y-axis title for ratio
            
        Returns:
            Styled ratio histogram
        """
        # Create unique name to avoid ROOT caching issues
        import time
        timestamp = str(int(time.time() * 1000000))
        
        # Create ratio histogram with unique name
        ratio_hist = numerator.Clone(f"ratio_{timestamp}_{name}")
        
        # Perform division with protection against extreme values
        ratio_hist.Divide(denominator)
        
        # Cap extreme ratio values to prevent y-axis scaling issues
        for i in range(1, ratio_hist.GetNbinsX() + 1):
            ratio_val = ratio_hist.GetBinContent(i)
            ratio_err = ratio_hist.GetBinError(i)
            
            # If ratio is extreme (> 10 or < 0.1), cap it or set to 0
            if ratio_val > 10.0:
                ratio_hist.SetBinContent(i, 0.0)
                ratio_hist.SetBinError(i, 0.0)
            elif ratio_val < 0.1 and ratio_val > 0:
                ratio_hist.SetBinContent(i, 0.0)
                ratio_hist.SetBinError(i, 0.0)
        
        # Style ratio
        ratio_hist.SetMarkerStyle(20)
        ratio_hist.SetMarkerSize(1.0)
        ratio_hist.SetLineColor(ROOT.kBlack)
        ratio_hist.SetMarkerColor(ROOT.kBlack)
        ratio_hist.SetLineStyle(1)
        ratio_hist.SetStats(0)  # Disable statistics box
        
        # Set axis properties for ratio
        ratio_hist.GetXaxis().SetTitle("R_{S}" if "ms" in name.lower() else "M_{S} [TeV]")
        ratio_hist.GetYaxis().SetTitle(ratio_title)
        ratio_hist.GetYaxis().SetRangeUser(0.5, 1.5)
        ratio_hist.GetXaxis().SetTitleSize(0.15)
        ratio_hist.GetYaxis().SetTitleSize(0.15)
        ratio_hist.GetXaxis().SetLabelSize(0.18)
        ratio_hist.GetXaxis().SetLabelOffset(0.02)
        ratio_hist.GetYaxis().SetLabelSize(0.12)
        ratio_hist.GetYaxis().SetTitleOffset(0.37)
        ratio_hist.GetXaxis().SetTitleOffset(1.25)
        ratio_hist.GetYaxis().SetNdivisions(505)
        ratio_hist.GetXaxis().CenterTitle()
        ratio_hist.GetYaxis().CenterTitle()
        
        return ratio_hist

    def create_datamc_ratio_canvas(self, data_hist: ROOT.TH1D, mc_histograms: List[Tuple], 
                                  group_labels: List[str], name: str, normalize: bool = False, 
                                  final_state: str = None) -> ROOT.TCanvas:
        """
        Create a two-pad canvas with stacked MC backgrounds and data/MC ratio.
        
        Args:
            data_hist: Data histogram
            mc_histograms: List of tuples (histogram, label) for MC backgrounds
            group_labels: Group labels for the plot
            name: Canvas name
            normalize: Whether to normalize histograms
            final_state: Final state name for SV label (optional)
            
        Returns:
            Canvas with two pads and Data/MC comparison
        """
        if not mc_histograms:
            raise ValueError("No MC histograms provided")
        
        # Normalize data histogram if requested
        if normalize:
            from unrolled_histogram_maker import UnrolledHistogramMaker
            hist_maker = UnrolledHistogramMaker()
            data_hist = hist_maker.normalize_histogram(data_hist, 'unity')
        
        # Use base ratio canvas setup
        canvas, pad1, pad2 = self._create_base_ratio_canvas(f"{name}_datamc", normalize)
        
        # === Top pad: Stacked distributions ===
        pad1.cd()
        
        # Use pre-registered custom colors
        mc_colors = self.custom_colors
        
        # Sort MC backgrounds by total yield (lowest to highest for proper stacking)
        mc_with_integrals = []
        for i, (mc_hist, label) in enumerate(mc_histograms):
            integral = mc_hist.Integral()
            mc_with_integrals.append((integral, mc_hist, label, i))
        
        # Sort by integral (lowest first)
        mc_with_integrals.sort(key=lambda x: x[0])
        
        print("MC backgrounds ordered by yield:")
        for integral, mc_hist, label, orig_idx in mc_with_integrals:
            print(f"  {label}: {integral:.1f}")
        
        # Create THStack for MC backgrounds
        stack = ROOT.THStack("stack", "")
        total_mc = None
        
        # First pass: create total MC sum before normalization
        for stack_idx, (integral, mc_hist, label, orig_idx) in enumerate(mc_with_integrals):
            if total_mc is None:
                total_mc = mc_hist.Clone("total_mc")
            else:
                total_mc.Add(mc_hist)
        
        # Apply normalization to total MC if requested
        if normalize:
            from unrolled_histogram_maker import UnrolledHistogramMaker
            hist_maker = UnrolledHistogramMaker()
            total_mc_normalized = hist_maker.normalize_histogram(total_mc, 'unity')
            
            # Calculate normalization factors for each group
            normalization_factors = []
            for group in range(3):  # 3 groups of 3 bins each
                group_sum_original = 0
                group_sum_normalized = 0
                for bin_idx in range(group * 3 + 1, (group + 1) * 3 + 1):
                    group_sum_original += total_mc.GetBinContent(bin_idx)
                    group_sum_normalized += total_mc_normalized.GetBinContent(bin_idx)
                
                if group_sum_original > 0:
                    normalization_factors.append(group_sum_normalized / group_sum_original)
                else:
                    normalization_factors.append(1.0)
            
            total_mc = total_mc_normalized
        
        # Second pass: style MC histograms and apply same normalization
        for stack_idx, (integral, mc_hist, label, orig_idx) in enumerate(mc_with_integrals):
            # Apply same normalization as total MC
            if normalize:
                for group in range(3):
                    factor = normalization_factors[group]
                    for bin_idx in range(group * 3 + 1, (group + 1) * 3 + 1):
                        old_content = mc_hist.GetBinContent(bin_idx)
                        old_error = mc_hist.GetBinError(bin_idx)
                        mc_hist.SetBinContent(bin_idx, old_content * factor)
                        mc_hist.SetBinError(bin_idx, old_error * factor)
            
            color = mc_colors[orig_idx % len(mc_colors)]
            print(f"Adding to stack: {label} (yield: {integral:.1f}, color: {color})")
            mc_hist.SetFillColor(color)
            mc_hist.SetLineColor(ROOT.kBlack)
            mc_hist.SetLineWidth(1)
            mc_hist.SetLineStyle(1)
            mc_hist.SetFillStyle(1001)
            stack.Add(mc_hist)
        
        # Set axis ranges
        if normalize:
            # For normalized plots, max is 1.0 by definition, use log scale
            stack.SetMinimum(0.001)  # Small value for log scale to handle zeros
            stack.SetMaximum(15.)     # 1.0 + 50% headroom
        else:
            # For non-normalized plots, calculate from actual data
            max_val = max(data_hist.GetMaximum(), stack.GetMaximum())
            stack.SetMinimum(0.5)  # For log scale
            stack.SetMaximum(max_val * 1.5)
        
        # Draw axis and grid first, then histogram without redrawing axis
        stack.Draw("AXIS")  # Draw only the axis frame and tick marks
        stack.GetXaxis().SetLabelSize(0)  # Hide x-labels on top pad
        
        # Set appropriate y-axis title based on normalization
        if normalize:
            stack.GetYaxis().SetTitle("normalized events")
        else:
            stack.GetYaxis().SetTitle("number of events")
            
        stack.GetYaxis().CenterTitle()
        stack.GetYaxis().SetTitleSize(0.06)
        stack.GetYaxis().SetLabelSize(0.05)
        pad1.Update()  # Force update to establish axis
        pad1.RedrawAxis("G")  # Draw grid lines behind everything
        stack.Draw("HIST SAME")  # Draw histogram content without redrawing axis
        
        # Create and style MC uncertainty band
        mc_uncertainty = total_mc.Clone("mc_uncertainty")
        mc_uncertainty.SetFillStyle(3244)  # Hatched pattern
        mc_uncertainty.SetFillColor(ROOT.kBlack)
        mc_uncertainty.SetLineColor(ROOT.kBlack)
        mc_uncertainty.SetLineWidth(1)  # Box border for legend
        mc_uncertainty.SetLineStyle(1)  # Solid line for legend border
        mc_uncertainty.Draw("E2 SAME")  # E2 = error band
        
        # Style data histogram
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerSize(1.0)
        data_hist.SetLineColor(ROOT.kBlack)
        data_hist.SetMarkerColor(ROOT.kBlack)
        data_hist.SetLineStyle(1)
        data_hist.Draw("PEX0 SAME")  # PE0 = markers with vertical error bars only
        
        # === Bottom pad: Ratio ===
        pad2.cd()
        
        # Use helper method to create ratio histogram
        ratio_hist = self._create_ratio_histogram(data_hist, total_mc, name, "#frac{data}{model}")
        
        # Set bin labels on ratio plot
        for i in range(1, ratio_hist.GetNbinsX() + 1):
            if i <= data_hist.GetNbinsX():
                ratio_hist.GetXaxis().SetBinLabel(i, data_hist.GetXaxis().GetBinLabel(i))
        
        ratio_hist.Draw("PEX0")  # PE0 = markers with vertical error bars only
        
        # Reference line at 1
        x_min = ratio_hist.GetXaxis().GetXmin()
        x_max = ratio_hist.GetXaxis().GetXmax()
        line = ROOT.TLine(x_min, 1, x_max, 1)
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kBlack)
        line.Draw()
        
        # === Create external legend ===
        canvas.cd()
        
        # Legend positioned in the right margin
        legend = ROOT.TLegend(0.8, 0.6, 1.05, 0.95)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.035)
        legend.SetMargin(0.15)
        #legend.SetEntrySeparation(1.)
        
        # Add data entry
        legend.AddEntry(data_hist, "data", "pey0")
        
        # Add MC uncertainty band
        legend.AddEntry(mc_uncertainty, "total uncertainty", "f")
        
        # Add MC backgrounds (reverse stack order for legend - highest yield first)
        for integral, mc_hist, label, orig_idx in reversed(mc_with_integrals):
            clean_label = self._clean_mc_label(label)
            legend.AddEntry(mc_hist, clean_label, "f")
        
        legend.Draw()
        
        # === Add decorations covering both pads ===
        canvas.cd()  # Draw overlay on main canvas, not just top pad
        
        # Add separator lines and group labels (adapted for two-pad layout)
        # Overlay covers entire canvas area including both pads
        overlay = ROOT.TPad("overlay", "overlay", 0, 0, 1, 1)
        overlay.SetFillStyle(0)
        overlay.SetFrameFillStyle(0)
        overlay.SetBorderSize(0)
        overlay.SetMargin(0, 0, 0, 0)
        overlay.Draw()
        overlay.cd()
        
        # Add separator lines to the top pad
        separator_lines = self._add_separator_lines_datamc(overlay, data_hist, pad1)
        
        # Add group labels at top of distribution pad
        self._add_group_labels_datamc(overlay, group_labels, pad1)
        
        # Add CMS labels
        cms_objects = self._add_cms_labels_datamc(overlay)
        
        # Add SV label if final state is provided
        sv_object = None
        if final_state:
            sv_object = self._add_sv_label_datamc(overlay, final_state)
        
        # Store objects to prevent garbage collection
        canvas.pad1 = pad1
        canvas.pad2 = pad2
        canvas.stack = stack
        canvas.total_mc = total_mc
        canvas.mc_uncertainty = mc_uncertainty
        canvas.ratio_hist = ratio_hist
        canvas.line = line
        canvas.legend = legend
        canvas.overlay = overlay
        canvas.data_hist = data_hist
        canvas.mc_histograms = mc_histograms
        canvas.mc_with_integrals = mc_with_integrals
        canvas.separator_lines = separator_lines
        canvas.cms_objects = cms_objects
        if sv_object:
            canvas.sv_object = sv_object
        
        canvas.Update()
        return canvas
    
    def create_postfit_ratio_canvas(self, data_hist: ROOT.TH1D, postfit_hist: ROOT.TH1D, 
                                   group_labels: List[str], name: str, normalize: bool = False) -> ROOT.TCanvas:
        """Simple postfit vs data comparison."""
        
        # Extract values and create fresh histograms
        postfit_values = []
        postfit_errors = []
        for i in range(1, postfit_hist.GetNbinsX() + 1):
            postfit_values.append(postfit_hist.GetBinContent(i))
            postfit_errors.append(postfit_hist.GetBinError(i))
        
        data_values = []
        data_errors = []
        for i in range(1, data_hist.GetNbinsX() + 1):
            data_values.append(data_hist.GetBinContent(i))
            data_errors.append(data_hist.GetBinError(i))
        
        # Create fresh histograms
        fresh_postfit = ROOT.TH1D(f"postfit_{name}", "Post-fit", 9, 0, 9)
        fresh_data = ROOT.TH1D(f"data_{name}", "Data", 9, 0, 9)

        fresh_postfit.SetDirectory(0)
        fresh_data.SetDirectory(0)
        fresh_postfit.SetStats(0)
        fresh_data.SetStats(0)
        
        # Store immediately to prevent garbage collection
        ROOT.SetOwnership(fresh_postfit, False)
        ROOT.SetOwnership(fresh_data, False)
        
        # Set bin labels based on grouping type
        if "ms" in name.lower():
            # MS grouping: RS labels for each bin
            bin_labels = ["[0.15,0.3]", "[0.3,0.4]", "[0.4,inf]"] * 3  
        else:
            # RS grouping: MS labels for each bin  
            bin_labels = ["[1.0,2.0]", "[2.0,3.0]", "[3.0,inf]"] * 3
            
        for i, label in enumerate(bin_labels):
            fresh_postfit.GetXaxis().SetBinLabel(i + 1, label)
            fresh_data.GetXaxis().SetBinLabel(i + 1, label)
        
        for i, (pval, perr, dval, derr) in enumerate(zip(postfit_values, postfit_errors, data_values, data_errors)):
            fresh_postfit.SetBinContent(i + 1, pval)
            fresh_postfit.SetBinError(i + 1, perr)
            fresh_data.SetBinContent(i + 1, dval)
            fresh_data.SetBinError(i + 1, derr)
            
        
        # Use base ratio canvas setup like datamc
        canvas, pad1, pad2 = self._create_base_ratio_canvas(f"{name}_postfit", normalize)
        
        # === Top pad: Distribution comparison (copy from datamc) ===
        pad1.cd()
        
        # Set axis ranges with increased maximum
        max_val = max(fresh_data.GetMaximum(), fresh_postfit.GetMaximum())
        fresh_postfit.SetMinimum(0.5)
        fresh_postfit.SetMaximum(max_val * 5.0)  # Increased from 1.5 to 5.0 for more space
        
        # Draw axis first like datamc function
        fresh_postfit.Draw("AXIS")
        fresh_postfit.GetXaxis().SetLabelSize(0)  # Hide x-labels on top pad
        
        # Set appropriate y-axis title
        if normalize:
            fresh_postfit.GetYaxis().SetTitle("normalized events")
        else:
            fresh_postfit.GetYaxis().SetTitle("number of events")
            
        fresh_postfit.GetYaxis().CenterTitle()
        fresh_postfit.GetYaxis().SetTitleSize(0.06)
        fresh_postfit.GetYaxis().SetLabelSize(0.05)
        pad1.Update()  # Force update to establish axis
        pad1.RedrawAxis("G")  # Draw grid lines behind everything
        
        # Style and draw postfit
        # Use a fixed transparent orange color index or create once
        if not hasattr(ROOT, '_transparent_orange_index'):
            ROOT._transparent_orange_index = ROOT.TColor.GetColorTransparent(ROOT.kOrange+7, 0.7)
        fresh_postfit.SetFillColor(ROOT._transparent_orange_index)
        fresh_postfit.SetLineColor(ROOT.kBlack)
        fresh_postfit.SetLineWidth(1)
        fresh_postfit.SetLineStyle(1)
        fresh_postfit.SetFillStyle(1001)  # Solid fill
        fresh_postfit.Draw("HIST SAME")
        
        # Create and style uncertainty band like datamc
        postfit_uncertainty = fresh_postfit.Clone("postfit_uncertainty")
        postfit_uncertainty.SetDirectory(0)
        postfit_uncertainty.SetFillStyle(3244)  # Hatched pattern
        postfit_uncertainty.SetFillColor(ROOT.kBlack)
        postfit_uncertainty.SetLineColor(ROOT.kBlack)
        postfit_uncertainty.SetLineWidth(1)  # Box border for legend
        postfit_uncertainty.SetLineStyle(1)  # Solid line for legend border
        postfit_uncertainty.Draw("E2 SAME")  # E2 = error band
        
        # Style and draw data  
        fresh_data.SetMarkerStyle(20)
        fresh_data.SetMarkerSize(1.0)
        fresh_data.SetLineColor(ROOT.kBlack)
        fresh_data.SetMarkerColor(ROOT.kBlack)
        fresh_data.SetLineStyle(1)
        fresh_data.Draw("PEX0 SAME")
        
        # === Bottom pad: Ratio (copy from datamc) ===
        pad2.cd()
        
        # Create ratio histogram
        ratio_hist = fresh_data.Clone(f"ratio_{name}")
        ratio_hist.SetDirectory(0)
        ratio_hist.SetTitle("")  # Remove title to avoid "Data" label
        ratio_hist.Divide(fresh_postfit)
        
        # Style ratio like datamc
        ratio_hist.SetMarkerStyle(20)
        ratio_hist.SetMarkerSize(1.0)
        ratio_hist.SetLineColor(ROOT.kBlack)
        ratio_hist.SetMarkerColor(ROOT.kBlack)
        ratio_hist.SetLineStyle(1)
        
        # Set axis properties like datamc
        ratio_hist.GetXaxis().SetTitle("R_{S}" if "ms" in name.lower() else "M_{S} [TeV]")
        ratio_hist.GetYaxis().SetTitle("#frac{data}{post-fit}")
        ratio_hist.GetYaxis().SetRangeUser(0.5, 1.5)
        ratio_hist.GetXaxis().SetTitleSize(0.15)
        ratio_hist.GetYaxis().SetTitleSize(0.15)
        ratio_hist.GetXaxis().SetLabelSize(0.18)
        ratio_hist.GetXaxis().SetLabelOffset(0.02)
        ratio_hist.GetYaxis().SetLabelSize(0.12)
        ratio_hist.GetYaxis().SetTitleOffset(0.37)
        ratio_hist.GetXaxis().SetTitleOffset(1.25)
        ratio_hist.GetYaxis().SetNdivisions(505)
        ratio_hist.GetXaxis().CenterTitle()
        ratio_hist.GetYaxis().CenterTitle()
        
        # Set bin labels on ratio plot
        for i in range(1, ratio_hist.GetNbinsX() + 1):
            if i <= fresh_data.GetNbinsX():
                ratio_hist.GetXaxis().SetBinLabel(i, fresh_data.GetXaxis().GetBinLabel(i))
        
        ratio_hist.Draw("PEX0")
        
        # Reference line at 1
        x_min = ratio_hist.GetXaxis().GetXmin()
        x_max = ratio_hist.GetXaxis().GetXmax()
        line = ROOT.TLine(x_min, 1, x_max, 1)
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kBlack)
        line.Draw()
        
        # === Create external legend like datamc ===
        canvas.cd()
        
        legend = ROOT.TLegend(0.8, 0.7, 1.05, 0.9)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.035)
        legend.SetMargin(0.15)
        
        legend.AddEntry(fresh_data, "data", "pe")
        legend.AddEntry(postfit_uncertainty, "post-fit uncertainty", "f")
        legend.AddEntry(fresh_postfit, "post-fit", "f")
        legend.Draw()
        
        # === Add decorations like datamc ===
        canvas.cd()
        
        overlay = ROOT.TPad("overlay", "overlay", 0, 0, 1, 1)
        overlay.SetFillStyle(0)
        overlay.SetFrameFillStyle(0)
        overlay.SetBorderSize(0)
        overlay.SetMargin(0, 0, 0, 0)
        overlay.Draw()
        overlay.cd()
        
        separator_lines = self._add_separator_lines_datamc(overlay, fresh_data, pad1)
        self._add_group_labels_datamc(overlay, group_labels, pad1)
        cms_objects = self._add_cms_labels_datamc(overlay)
        
        # Store objects to prevent garbage collection
        canvas.pad1 = pad1
        canvas.pad2 = pad2
        canvas.fresh_postfit = fresh_postfit
        canvas.fresh_data = fresh_data
        canvas.postfit_uncertainty = postfit_uncertainty
        canvas.ratio_hist = ratio_hist
        canvas.line = line
        canvas.legend = legend
        canvas.overlay = overlay
        canvas.separator_lines = separator_lines
        canvas.cms_objects = cms_objects
        
        return canvas
    
    def _add_group_labels_datamc(self, overlay_pad: ROOT.TPad, group_labels: List[str], main_pad: ROOT.TPad) -> None:
        """Add group labels for data/MC canvas."""
        # For full canvas overlay, we need to map to the top pad area
        # Top pad is at (0, 0.3, 0.75, 1.0) in canvas coordinates
        pad_canvas_left = 0.0
        pad_canvas_right = 0.8  # 85% of canvas width
        pad_canvas_width = pad_canvas_right - pad_canvas_left
        
        # Account for margins within the top pad
        pad_left_margin = main_pad.GetLeftMargin() 
        pad_right_margin = main_pad.GetRightMargin()
        
        # Effective plotting area within the top pad
        plot_left = pad_canvas_left + pad_left_margin * pad_canvas_width
        plot_right = pad_canvas_right - pad_right_margin * pad_canvas_width
        plot_width = plot_right - plot_left
        
        # Each group covers 3 bins
        group_width = plot_width / 3
        
        latex = ROOT.TLatex()
        latex.SetTextAlign(22)
        latex.SetTextSize(0.035)
        latex.SetTextFont(42)
        latex.SetNDC(True)
        
        for i, group_name in enumerate(group_labels):
            # Position in canvas coordinates, mapped to top pad area
            group_center = plot_left + group_width * (i + 0.5)
            y_pos = 0.91  # Updated position for group labels
            latex.DrawLatex(group_center, y_pos, group_name)
    
    def _add_separator_lines_datamc(self, overlay_pad: ROOT.TPad, hist: ROOT.TH1D, main_pad: ROOT.TPad) -> List[ROOT.TLine]:
        """Add separator lines for data/MC canvas extending through both pads."""
        lines = []
        
        # For full canvas overlay, map to the plotting area
        # Top pad is at (0, 0.3, 0.75, 1.0) in canvas coordinates
        pad_canvas_left = 0.0
        pad_canvas_right = 0.8  # 85% of canvas width
        pad_canvas_width = pad_canvas_right - pad_canvas_left
        
        # Account for margins within the top pad
        pad_left_margin = main_pad.GetLeftMargin()
        pad_right_margin = main_pad.GetRightMargin()
        
        # Effective plotting area within the canvas coordinates
        plot_left = pad_canvas_left + pad_left_margin * pad_canvas_width
        plot_right = pad_canvas_right - pad_right_margin * pad_canvas_width
        plot_width = plot_right - plot_left
        
        # Convert x positions to canvas NDC coordinates
        x_axis = hist.GetXaxis()
        
        def x_bin_to_ndc(bin_index):
            # Convert bin edge to canvas NDC coordinates
            x_edge = x_axis.GetBinLowEdge(bin_index + 1)
            x_normalized = x_edge / x_axis.GetNbins()  # Normalize to [0,1] 
            return plot_left + x_normalized * plot_width
        
        # Draw vertical lines at bin boundaries 3 and 6 (between groups)
        # Draw from bottom of ratio pad to near top of distribution pad
        y_bottom = 0.065   # Bottom of canvas (bottom of ratio pad)
        y_top = 0.945     # Near top of distribution pad (below group labels)
        
        for bin_edge in [3, 6]:
            x_ndc = x_bin_to_ndc(bin_edge)
            line = ROOT.TLine()
            line.SetNDC(True)
            line.SetLineColor(self.label_config['separator_line_color'])
            line.SetLineWidth(self.label_config['separator_line_width'])
            line.SetLineStyle(1)
            line.DrawLine(x_ndc, y_bottom, x_ndc, y_top)
            lines.append(line)
        
        return lines
    
    def _add_cms_labels_datamc(self, overlay_pad: ROOT.TPad) -> List[ROOT.TLatex]:
        """Add CMS labels for data/MC canvas (without SV label)."""
        
        y_pos = 0.958
        
        # Use universal CMS mark
        cms_objects = self.universal_cms_mark(0.12, y_pos, 0.04)
        
        # Add luminosity label
        lumi_latex = ROOT.TLatex()
        lumi_latex.SetTextFont(42)
        lumi_latex.SetNDC()
        lumi_latex.SetTextSize(0.04)
        lumi_latex.SetTextAlign(31)  # Right align
        lumi_latex.DrawLatex(0.785, y_pos, f"{self.luminosity:.0f} fb^{{-1}} (13 TeV)")
        
        return cms_objects + [lumi_latex]
    
    def _add_sv_label_datamc(self, overlay_pad: ROOT.TPad, final_state: str, x_pos: float = 0.63, y_pos: float = 0.958):
        """Add SV label for data/MC canvas (separate from CMS labels)."""
        
        sv_label = self._format_sv_label(final_state)
        
        # Use TMathText for labels containing \\ell\\ell, otherwise use TLatex
        if "\\ell\\ell" in sv_label:
            ROOT.gEnv.SetValue("TMathText.FontResolution", 200)
            sv_text = ROOT.TMathText()
            sv_text.SetTextFont(42)
            sv_text.SetNDC()
            sv_text.SetTextSize(0.04)
            sv_text.SetTextAlign(31)  # Right align
            sv_text.DrawMathText(x_pos, y_pos, sv_label)
        else:
            sv_text = ROOT.TLatex()
            sv_text.SetTextFont(42)
            sv_text.SetNDC()
            sv_text.SetTextSize(0.04)
            sv_text.SetTextAlign(31)  # Right align
            sv_text.DrawLatex(x_pos, y_pos, sv_label)
        
        return sv_text 
        
    def set_canvas_config(self, **kwargs) -> None:
        """
        Update canvas configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key in self.canvas_config:
                self.canvas_config[key] = value
            elif key in self.label_config:
                self.label_config[key] = value
    
    def get_canvas_config(self) -> Dict:
        """Get current canvas configuration."""
        return {**self.canvas_config, **self.label_config}
