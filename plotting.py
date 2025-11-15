import ROOT
from typing import List
import numpy as np

class Plot:
    @staticmethod
    def CMSmark(plot_title: str = "",
                x: float = 0.12) -> None:
        # Official CMS formatting parameters
        cmsText = "CMS"
        cmsTextFont = 61
        extraText = "Preliminary"
        extraTextFont = 52
        cmsTextSize = 1.00
        cmsTextOffset = 0.01
        extraOverCmsTextSize = 0.78
        relPosX = 0.045
        relPosY = 0.025
        
        # Draw plot title if provided
        if plot_title:
            latex_title = ROOT.TLatex()
            latex_title.SetTextFont(42)
            latex_title.SetNDC()
            latex_title.SetTextSize(0.035)
            latex_title.DrawLatex(0.51, 0.91, plot_title)
        
        # Draw CMS text
        latex_cms = ROOT.TLatex()
        latex_cms.SetNDC()
        latex_cms.SetTextAlign(13)  # Left bottom align
        latex_cms.SetTextFont(cmsTextFont)
        latex_cms.SetTextSize(cmsTextSize * 0.04)  # Scale to reasonable size
        cms_x = x + cmsTextOffset
        cms_y = 0.91 + relPosY
        latex_cms.DrawLatex(cms_x, cms_y, cmsText)
        
        # Draw "Preliminary" text
        latex_extra = ROOT.TLatex()
        latex_extra.SetNDC()
        latex_extra.SetTextAlign(13)  # Left bottom align  
        latex_extra.SetTextFont(extraTextFont)
        latex_extra.SetTextSize(extraOverCmsTextSize * cmsTextSize * 0.04)
        extra_x = cms_x + relPosX
        extra_y = cms_y - 0.008
        latex_extra.DrawLatex(extra_x, extra_y, extraText)


    @staticmethod
    def plot_histogram1D(histogram: ROOT.TH1D,
                         name: str,
                         xlabel: str) -> ROOT.TCanvas:
        """
        Plot a single TH1D histogram on a canvas.
        
        Args:
            histogram: TH1D histogram to plot
            xlabel: Label for x-axis
        
        Returns:
            ROOT.TCanvas: The created canvas with the plotted histogram
        """
        # Create canvas
        canvas = ROOT.TCanvas(f"can_{name}_{histogram.GetName()}", "Histogram Plot", 800, 600)
        canvas.cd()
        
        # Set up canvas properties
        canvas.SetGrid()
        canvas.SetLogy()
        
        # Set up histogram properties
        histogram.SetStats(0)
        histogram.SetTitle("")
        histogram.GetXaxis().SetTitle(xlabel)
        histogram.GetYaxis().SetTitle("Entries")
        histogram.GetXaxis().CenterTitle(True)
        histogram.GetYaxis().CenterTitle(True)
    
        # Set histogram style
        histogram.SetLineWidth(3)
        histogram.SetLineColor(ROOT.kBlue + 2)
        
        # Set y-axis range
        max_y = histogram.GetMaximum()
        y_max = max_y * 1.2
        histogram.GetYaxis().SetRangeUser(0.01, y_max)
        
        # Draw histogram
        histogram.Draw("HIST")
        
        # Update canvas
        canvas.Update()
        
        # Draw CMS mark
        Plot.CMSmark()
        
        # Store histogram and force another update
        canvas.histogram = histogram
        canvas.Modified()
        canvas.Update()
        
        return canvas
        
    @staticmethod
    def plot_histograms(histograms: List[ROOT.TH1D],
                       labels: List[str], 
                       name: str, 
                       xlabel: str) -> ROOT.TCanvas:
        """
        Plot multiple TH1D histograms on the same canvas with a legend.
        
        Args:
            histograms: List of TH1D histograms
            labels: List of labels for the legend
            name: Name for the canvas
            xlabel: Label for x-axis
            
        Returns:
            ROOT.TCanvas: The created canvas with the plotted histograms
        """
        # Convert boost histograms to ROOT histograms
        root_hists = histograms

        # Create canvas first
        canvas = ROOT.TCanvas(f"can_{name}", "Histogram Plot", 800, 600)
        canvas.cd()
        
        # Set up canvas properties early
        canvas.SetGrid()
        canvas.SetLogy()
        
        # Create legend
        canvas.legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        canvas.legend.SetFillStyle(0)  # Make legend transparent
        canvas.legend.SetBorderSize(0)  # Make legend border transparent
        
        # Find maximum y-value first
        max_y = max(h.GetMaximum() for h in root_hists)
        y_max = max_y * 1.2
         
        # Define colors
        colors = [
            ROOT.kBlue + 2,
            ROOT.kGreen + 2,
            ROOT.kRed + 2,
            ROOT.kOrange - 3,
            ROOT.kMagenta - 2,
            ROOT.kBlack,
            ROOT.kAzure + 2,
            ROOT.kGray
        ]
        
        # Set up first histogram
        root_hists[0].SetStats(0)
        root_hists[0].SetTitle("")
        root_hists[0].GetXaxis().SetTitle(xlabel)
        root_hists[0].GetXaxis().SetTitleOffset(1.2)
        root_hists[0].GetYaxis().SetTitle("Events")
        root_hists[0].GetYaxis().SetRangeUser(0.01, y_max)
        root_hists[0].GetXaxis().CenterTitle(True)
        root_hists[0].GetYaxis().CenterTitle(True)
        
        # Plot histograms with explicit options
        for i, hist in enumerate(root_hists):
            hist.SetLineWidth(3)
            hist.SetLineColor(colors[i])
            
            if i == 0:
                hist.Draw("HIST")  # Draw first histogram with HIST option
                #print(f"Drawing first histogram {labels[i]} with integral: {hist.Integral()}")
            else:
                hist.Draw("HIST SAME")  # Draw others with HIST SAME
                #print(f"Drawing histogram {labels[i]} with integral: {hist.Integral()}")
            
            canvas.legend.AddEntry(hist, labels[i], "l")
        
        # Update canvas to ensure all histograms are drawn
        canvas.Update()
        
        # Draw legend and CMS mark
        canvas.legend.Draw()
        Plot.CMSmark()
        
        # Store histograms and force another update
        canvas.histograms = root_hists
        canvas.Modified()
        canvas.Update()
        
        return canvas

    @staticmethod
    def plot_histogram(hist2D: ROOT.TH2D,
                       name: str,
                       xlabel: str,
                       ylabel: str,
                       zlabel: str = "Events") -> ROOT.TCanvas:
        """
        Plot a 2D histogram with color scale.
        
        Args:
            hist2D: ROOT TH2D or boost 2D histogram
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            zlabel: Label for z-axis (color scale)
            
        Returns:
            ROOT.TCanvas: The created canvas with the plotted histogram
        """
        root_hist = hist2D

        # Get histogram name
        #name = root_hist.GetName()
        
        # Create canvas
        canvas = ROOT.TCanvas(f"{name}_canvas", f"{name}_canvas", 800, 600)
        
        # Set histogram properties
        root_hist.SetStats(0)
        root_hist.SetTitle("")
        
        # Set axis labels
        root_hist.GetXaxis().SetTitle(xlabel)
        root_hist.GetYaxis().SetTitle(ylabel)
        root_hist.GetZaxis().SetTitle(zlabel)
        root_hist.GetXaxis().CenterTitle(True)
        root_hist.GetYaxis().CenterTitle(True)
        root_hist.GetZaxis().CenterTitle(True)
        root_hist.GetXaxis().SetTitleOffset(1.2)  # Offset bottom label
        
        # Set canvas properties
        canvas.SetGridx()
        canvas.SetGridy()
        canvas.SetLogz()
        canvas.SetLeftMargin(0.12)
        canvas.SetBottomMargin(0.12)
        canvas.SetRightMargin(0.15)
        
        # Draw histogram with color scale
        root_hist.Draw("colz")
        
        # Add CMS mark
        Plot.CMSmark()
        
        # Store histogram as canvas attribute to prevent garbage collection
        canvas.histogram = root_hist
        
        return canvas

    @staticmethod
    def plot_efficiency(efficiencies: List[ROOT.TEfficiency],
                       labels: List[str],
                       name: str,
                       xlabel: str) -> ROOT.TCanvas:
        """
        Plot multiple efficiency graphs on the same canvas.
        
        Args:
            efficiencies: List of ROOT TEfficiency objects
            labels: List of labels for the legend
            name: Name for the canvas
            xlabel: Label for x-axis
            
        Returns:
            ROOT.TCanvas: The created canvas with the plotted efficiencies
        """
        # Define markers and colors
        markers = [20, 21, 22, 23, 29, 33, 34, 43, 49]
        colors = [
            ROOT.kBlue + 2,
            ROOT.kGreen + 2,
            ROOT.kRed + 2,
            ROOT.kOrange - 3,
            ROOT.kMagenta - 2,
            ROOT.kBlack,
            ROOT.kAzure + 2,
            ROOT.kGray
        ]
        
        # Create canvas and multigraph
        canvas = ROOT.TCanvas(f"can_{name}", f"can_{name}", 800, 600)
        canvas.multigraph = ROOT.TMultiGraph()  # Store as canvas attribute to prevent garbage collection
        
        # Create and configure legend
        canvas.legend = ROOT.TLegend(0.5, 0.8, 0.8, 0.89)
        canvas.legend.SetFillStyle(0)  # Make legend transparent
        canvas.legend.SetBorderSize(0)  # Make legend border transparent
        
        # Set canvas properties
        canvas.SetLeftMargin(0.15)
        canvas.SetBottomMargin(0.12)
        canvas.SetGrid()
        
        # Process each efficiency
        for i, eff in enumerate(efficiencies):
            eff.SetMarkerStyle(markers[i])
            eff.SetStatisticOption(ROOT.TEfficiency.kFNormal)
            eff.SetMarkerColor(colors[i])
            eff.SetLineColor(colors[i])
            
            # Add to multigraph
            graph = eff.CreateGraph()
            canvas.multigraph.Add(graph)
            
            # Add to legend
            canvas.legend.AddEntry(eff, labels[i], "lep")
        
        # Draw multigraph
        canvas.multigraph.Draw("ap")
        
        # Set axis properties
        canvas.multigraph.SetMinimum(0.)
        canvas.multigraph.SetMaximum(1.)
        canvas.multigraph.GetXaxis().SetTitle(xlabel)
        canvas.multigraph.GetXaxis().SetTitleOffset(1.25)
        canvas.multigraph.GetXaxis().CenterTitle(True)
        canvas.multigraph.GetYaxis().SetTitle("Efficiency")
        canvas.multigraph.GetYaxis().CenterTitle(True)
        
        # Draw legend
        canvas.legend.Draw("same")
        
        # Update pad
        ROOT.gPad.Update()
        
        # Add CMS mark
        Plot.CMSmark(x=0.16)
        
        # Store efficiencies as canvas attribute to prevent garbage collection
        canvas.efficiencies = efficiencies
        
        return canvas

    @staticmethod
    def plot_histogram2D_division(numerator: ROOT.TH2D,
                                  denominator: ROOT.TH2D,
                                  name: str,
                                  xlabel: str,
                                  ylabel: str,
                                  title: str = "Ratio") -> ROOT.TCanvas:
        """
        Divide two TH2D histograms and plot the result on a canvas.
    
        Args:
            numerator: TH2D histogram for the numerator
            denominator: TH2D histogram for the denominator
            name: Name identifier for the canvas
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            title: Title for the z-axis (default: "Efficiency")
        
        Returns:
            ROOT.TCanvas: The created canvas with the plotted ratio histogram
        """
        # Create a copy of the numerator for the ratio
        ratio = numerator.Clone(f"ratio_{name}")
        ratio.SetTitle("")
        
        # Perform the division with binomial errors
        ratio.Divide(numerator, denominator, 1.0, 1.0, "B")
        
        # Create canvas
        canvas = ROOT.TCanvas(f"can_{name}_ratio", "2D Histogram Division", 900, 700)
        canvas.cd()
        
        # Set up canvas properties
        canvas.SetGrid()
        canvas.SetLeftMargin(0.12)
        canvas.SetBottomMargin(0.12)
        canvas.SetRightMargin(0.15)  # Make room for color palette
        
        # Set up histogram properties
        ratio.SetStats(0)
        ratio.GetXaxis().SetTitle(xlabel)
        ratio.GetYaxis().SetTitle(ylabel)
        ratio.GetZaxis().SetTitle(title)
        ratio.GetXaxis().CenterTitle(True)
        ratio.GetYaxis().CenterTitle(True)
        ratio.GetZaxis().CenterTitle(True)
        ratio.GetXaxis().SetTitleOffset(1.2)  # Offset bottom label
        
        # Set z-axis range (0 to 1 for efficiency plots)
        #ratio.GetZaxis().SetRangeUser(0.0, 1.0)
        
        # Set histogram style
        ratio.SetMarkerSize(0.8)
        
        # Draw histogram with color palette
        ratio.Draw("COLZ")
        
        # Update canvas
        canvas.Update()
        
        # Draw CMS mark
        Plot.CMSmark()
        
        # Store histograms and force another update
        canvas.ratio = ratio
        canvas.numerator = numerator
        canvas.denominator = denominator
        canvas.Modified()
        canvas.Update()
    
        return canvas
    
class EfficiencyPlotter:
    def __init__(self, histograms):
        self.histograms = histograms
        self.efficiency_configs = {}
        self.efficiencies = {}

    def add_base_config(self, key, label, denominator, efftag="GenParticleDxy"):
        """Add a base configuration for efficiency."""
        self.efficiency_configs[key] = {
            "numerator": f"{efftag}_{key}",
            "denominator": denominator,
            "label": label,
            "efftag": efftag
        }

    def add_cut_configs(self, base_key, cuts):
        """Add cut-based configurations for a base efficiency."""
        base = self.efficiency_configs[base_key]
        efftag = base["efftag"]
        for cut_key, cut in cuts.items():
            cut_key_full = f"{base_key}{cut_key}"
            self.efficiency_configs[cut_key_full] = {
                "numerator": f"{efftag}_{base_key}{cut['suffix']}",
                "denominator": base["denominator"],
                "label": cut["label"],
                "efftag": efftag
            }

    def create_efficiencies(self):
        """Generate TEfficiencies based on the configurations."""
        for key, config in self.efficiency_configs.items():
            try:
                num_hist = self.histograms[config["numerator"]]
                den_hist = self.histograms[config["denominator"]]
                self.efficiencies[key] = ROOT.TEfficiency(num_hist, den_hist)
            except KeyError as e:
                print(f"Warning: Could not create efficiency for {key}. Missing histogram: {e}")
                continue

    def create_group(self, keys, canvas_name, x_axis_label):
        """Create a group of efficiencies for plotting."""
        efficiencies = []
        labels = []
    
        for key in keys:
            if key in self.efficiencies:
                efficiencies.append(self.efficiencies[key])
                labels.append(self.efficiency_configs[key]["label"])
            else:
                print(f"Warning: No efficiency found for {key}")
    
        if not efficiencies:
            raise ValueError(f"No valid efficiencies found for group {canvas_name}")
        
        return {
            "efficiencies": efficiencies,
            "labels": labels,
            "canvas_name": canvas_name,
            "x_axis_label": x_axis_label
        }
    
    def plot_groups(self, groups):
        """Plot all the defined groups and return canvases."""
        return {
            group_config["canvas_name"]: Plot.plot_efficiency(
                group_config["efficiencies"],
                group_config["labels"],
                group_config["canvas_name"],
                group_config["x_axis_label"]
            )
            for group_config in groups
        }
