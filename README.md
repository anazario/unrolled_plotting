# Unrolled Plotting Framework

A comprehensive ROOT/PyROOT-based framework for creating unrolled 2D→1D histogram analysis plots for particle physics data analysis, particularly designed for SUSY searches and similar analyses.

## Overview

This framework transforms 2D histograms into 1D "unrolled" plots by mapping 2D bins into sequential 1D bins, organized into meaningful groups. It supports various plot types including single plots, comparison plots, marker-based jittered plots, and Data/MC ratio plots with stacked backgrounds.

## Features

- **Unrolled 2D→1D Analysis**: Convert 2D kinematic distributions into 1D plots for better visualization
- **Multiple Plot Types**: Single plots, multi-final-state comparisons, marker plots with jittering, Data/MC ratios
- **Flexible Grouping**: Support for both mass-based (Ms) and R-parameter-based (Rs) groupings
- **Data/MC Comparisons**: Stacked background plots with data overlay and ratio panels
- **Normalization Options**: Support for unity normalization and other normalization schemes
- **High-Quality Output**: Multiple output formats (PDF, PNG, ROOT, EPS, SVG) with publication-quality styling
- **Uncertainty Visualization**: Hatched uncertainty bands for total MC uncertainties
- **Interactive Analysis**: Both CLI and Python API interfaces

## Installation & Requirements

### Prerequisites
- ROOT (CERN) with PyROOT bindings
- Python 3.7+
- Required Python packages:
  ```bash
  pip install uproot numpy
  ```

### Setup
```bash
git clone <your-repository>
cd unrolled_plotting
```

## Framework Architecture

### Core Components

1. **`UnrolledDataProcessor`**: Handles ROOT file loading, event selection, and data preprocessing
2. **`UnrolledHistogramMaker`**: Creates and styles histograms with proper binning and labels
3. **`UnrolledCanvasMaker`**: Manages canvas creation, plot layout, and visual styling
4. **`UnrolledPlotter`**: High-level interface coordinating all components

### Key Concepts

- **Unrolling**: 2D (x,y) histograms are transformed into 1D by sequential bin mapping
- **Grouping**: Bins are organized into groups of 3, representing different kinematic regions
- **Final States**: Different physics selections (e.g., `passSelHighDM`, `passSelLowDM`)
- **Baseline Cuts**: Standard event selection criteria applied to all plots

## CLI Usage

### Basic Commands

#### Single Plot Creation
```bash
python unrolled_plotter.py \
    --files data.root \
    --final-state passSelHighDM \
    --grouping Ms \
    --output-file my_plot \
    --output-formats pdf png
```

#### Multi Final State Comparison
```bash
python unrolled_plotter.py \
    --files signal.root \
    --final-states passSelHighDM passSelLowDM passSelHighDM_bin1 \
    --grouping Rs \
    --output-file comparison_plot \
    --output-formats pdf png
```

#### Marker-Based Jittered Plots
```bash
python unrolled_plotter.py \
    --files file1.root file2.root file3.root \
    --final-state passSelHighDM \
    --grouping Ms \
    --markers \
    --output-file marker_plot \
    --output-formats pdf
```

#### Data/MC Ratio Plots
```bash
python unrolled_plotter.py \
    --data data_2018.root \
    --background QCD.root WJets.root TTJets.root \
    --final-states passSelHighDM passSelLowDM \
    --labels "QCD multijets" "W + jets" "t#bar{t} + jets" \
    --output-file datamc_comparison \
    --output-formats pdf png
```

### Advanced Options

#### Normalization
```bash
# Normalize histograms for shape comparison
python unrolled_plotter.py \
    --files signal1.root signal2.root \
    --final-state passSelHighDM \
    --normalize \
    --output-file normalized_comparison
```

#### Signal Scaling
```bash
# Scale signal by factor of 100 for visibility
python unrolled_plotter.py \
    --files signal.root background.root \
    --final-state passSelHighDM \
    --signal-scale 100 \
    --output-file scaled_signal
```

#### Background Scaling
```bash
# Scale backgrounds in Data/MC plots
python unrolled_plotter.py \
    --data data.root \
    --background mc1.root mc2.root \
    --background-scale 1.2 \
    --output-file scaled_datamc
```

#### Custom Output Organization
```bash
# Save all formats to shared folder
python unrolled_plotter.py \
    --files data.root \
    --final-state passSelHighDM \
    --output-file my_analysis \
    --output-formats pdf png eps svg  # All non-ROOT formats go to my_analysis/ folder
```

## Python API Usage

### Basic Single Plot

```python
from unrolled_plotter import UnrolledPlotter

# Initialize plotter
plotter = UnrolledPlotter(
    grouping_type='Ms',  # or 'Rs'
    luminosity=137.2     # fb^-1
)

# Create single plot
result = plotter.create_single_plot(
    file_path='data.root',
    plot_type='signal',
    final_state_flags=['passSelHighDM'],
    signal_scale=1.0,
    style_type='signal'
)

if result['success']:
    canvas = result['canvas']
    canvas.SaveAs('my_plot.pdf')
```

### Multi-Final-State Comparison

```python
# Create comparison plot
result = plotter.create_multi_final_state_plot(
    file_path='signal.root',
    final_state_flags=['passSelHighDM', 'passSelLowDM', 'passSelHighDM_bin1'],
    signal_scale=50.0,
    name='multi_fs_comparison',
    output_path='comparison',
    output_formats=['pdf', 'png'],
    normalize=False
)
```

### Marker-Based Plots

```python
# Create marker plot with jittering
result = plotter.create_multi_final_state_plot_with_markers(
    file_paths=['file1.root', 'file2.root', 'file3.root'],
    labels=['Sample 1', 'Sample 2', 'Sample 3'],
    final_state_flags=['passSelHighDM'],
    name='marker_comparison',
    output_path='markers',
    output_formats=['pdf'],
    normalize=True
)
```

### Data/MC Ratio Plots

```python
# Create Data/MC comparison
result = plotter.create_datamc_ratio_plot(
    data_file='data_2018.root',
    mc_files=['QCD.root', 'WJets.root', 'TTJets.root'],
    final_state_flags=['passSelHighDM'],
    mc_labels=['QCD multijets', 'W + jets', 't#bar{t} + jets'],
    data_scale=1.0,
    mc_scales=[1.0, 1.0, 1.0],
    name='datamc_ratio',
    output_path='datamc',
    output_formats=['pdf', 'png'],
    normalize=False
)
```

### Advanced Configuration

```python
# Custom canvas configuration
plotter.canvas_maker.set_canvas_config(
    width=1400,
    height=700,
    left_margin=0.15
)

# Custom histogram styling
plotter.histogram_maker.set_style_config(
    line_width=2,
    marker_size=1.2
)
```

## Plot Types & Features

### 1. Single Plots
- Individual histogram visualization
- Customizable styling and scaling
- Group labels and separation lines
- CMS preliminary labeling

### 2. Comparison Plots
- Multi-final-state overlays
- Automatic color assignment
- Legend management
- Optional normalization

### 3. Marker Plots
- Jittered marker positioning
- Multiple file comparison
- Error bar visualization
- Yield-based sorting

### 4. Data/MC Ratio Plots
- **Top pad**: Stacked MC backgrounds with data overlay
- **Bottom pad**: Data/MC ratio with reference line
- **Uncertainty bands**: Hatched total MC uncertainty
- **Legend**: External legend with proper ordering
- **Visual features**:
  - Grid lines behind histograms
  - Smaller tick marks
  - No x-error bars on data points
  - Proper axis labeling

## Configuration Files

### Baseline Event Selection
The framework applies standard baseline cuts:
- `selCMet>150`: Missing energy requirement
- `hlt_flags`: Trigger requirements  
- `cleaning_flags`: Data quality cuts
- `rjrCleaningVeto0`: Additional cleaning veto

### Grouping Configurations
- **Ms grouping**: Mass-based kinematic regions
- **Rs grouping**: R-parameter-based regions
- Each group contains 3 bins representing different kinematic ranges

## Output Formats

### Supported Formats
- **PDF**: Vector format, publication ready
- **PNG**: High-resolution raster (2x scaling with enhanced styling)
- **ROOT**: Native ROOT canvas format
- **EPS**: Encapsulated PostScript
- **SVG**: Scalable vector graphics

### File Organization
- **ROOT files**: Saved individually with canvas objects
- **Other formats**: Saved to shared folder named after output file
- **Folder structure**:
  ```
  output_name/
  ├── canvas1.pdf
  ├── canvas1.png
  ├── canvas2.pdf
  └── canvas2.png
  ```

## Troubleshooting

### Common Issues

1. **ROOT file access errors**: Check file paths and permissions
2. **Missing branches**: Verify ROOT file contains expected variables
3. **Empty histograms**: Check event selection criteria and final state flags
4. **Styling issues**: Ensure proper ROOT environment setup

### Debug Options
- Use `--verbose` flag for detailed output
- Check analysis summary for errors and statistics
- Verify baseline cuts are being applied correctly

## Advanced Features

### Custom Normalization
```python
# Access histogram maker directly for custom normalization
hist_maker = plotter.histogram_maker
normalized_hist = hist_maker.normalize_histogram(hist, 'unity')
```

### Canvas Customization
```python
# Direct canvas manipulation
canvas = plotter.canvas_maker.create_base_canvas('my_canvas')
# Add custom elements...
```

### Batch Processing
```python
# Process multiple files efficiently
file_list = ['file1.root', 'file2.root', 'file3.root']
for file_path in file_list:
    result = plotter.create_single_plot(
        file_path=file_path,
        plot_type='signal',
        # ... other parameters
    )
```

## Performance Considerations

- **Memory management**: Canvases store references to prevent garbage collection
- **File caching**: Multiple plots from same file use cached data
- **Parallel processing**: Multiple canvas creation can be done in parallel
- **Color registration**: Custom colors are pre-registered for efficiency

## Contributing

When modifying the framework:

1. **Maintain backwards compatibility** in API changes
2. **Update documentation** for new features
3. **Add error handling** for new functionality
4. **Test with various data formats** and final states
5. **Follow ROOT best practices** for memory management

## Example Analysis Workflow

```python
#!/usr/bin/env python3
"""
Example analysis script using unrolled plotting framework
"""
from unrolled_plotter import UnrolledPlotter

def main():
    # Initialize plotter for high-mass analysis
    plotter = UnrolledPlotter(
        grouping_type='Ms',
        luminosity=137.2
    )
    
    # Define analysis samples
    data_file = 'data_2018.root'
    background_files = [
        'QCD_HT.root',
        'WJets_HT.root', 
        'TTJets.root',
        'ZInv.root'
    ]
    signal_file = 'T2tt_500_320.root'
    
    # Final states to analyze
    final_states = [
        'passSelHighDM',
        'passSelLowDM', 
        'passSelHighDM_bin1',
        'passSelHighDM_bin2'
    ]
    
    # 1. Create Data/MC comparison
    print("Creating Data/MC ratio plots...")
    datamc_result = plotter.create_multi_datamc_ratio_plots(
        data_file=data_file,
        mc_files=background_files,
        final_state_flags=final_states,
        mc_labels=['QCD', 'W+jets', 't#bar{t}', 'Z→#nu#nu'],
        base_name='datamc_analysis',
        output_path='plots/datamc_comparison',
        output_formats=['pdf', 'png']
    )
    
    # 2. Create signal comparison
    print("Creating signal comparison plots...")
    signal_result = plotter.create_multi_final_state_plot(
        file_path=signal_file,
        final_state_flags=final_states,
        signal_scale=100,  # Scale up for visibility
        name='signal_comparison',
        output_path='plots/signal_shapes',
        output_formats=['pdf'],
        normalize=True  # Shape comparison
    )
    
    # 3. Create marker plot comparing different signal masses
    print("Creating signal mass comparison...")
    signal_files = [
        'T2tt_400_300.root',
        'T2tt_500_320.root',
        'T2tt_600_350.root'
    ]
    marker_result = plotter.create_multi_final_state_plot_with_markers(
        file_paths=signal_files,
        labels=['(400,300)', '(500,320)', '(600,350)'],
        final_state_flags=['passSelHighDM'],
        name='signal_mass_scan',
        output_path='plots/mass_comparison',
        output_formats=['pdf', 'png']
    )
    
    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"Data/MC plots: {'✓' if datamc_result['success'] else '✗'}")
    print(f"Signal shapes: {'✓' if signal_result['success'] else '✗'}")  
    print(f"Mass comparison: {'✓' if marker_result['success'] else '✗'}")

if __name__ == '__main__':
    main()
```

