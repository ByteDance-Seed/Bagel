# BAGEL Image Generation Comparison Tool

## Overview
A Streamlit application for visualizing and comparing generated images from BAGEL model evaluations. The app allows side-by-side comparison of different models, checkpoints, and hyperparameter configurations.

## Features

### 🚀 Improved Caching System
- **Persistent File-Based Cache**: Results are cached to disk (`.streamlit_cache/results_cache.pkl`) for fast loading
- **24-Hour Cache Validity**: Cache automatically expires after 24 hours
- **Manual Cache Control**: 
  - Force Refresh button to re-scan directories
  - Clear Cache button to remove cached data
- **Progress Tracking**: Visual progress bar during directory scanning
- **Cache Status Display**: Shows when cache was last updated

### 📊 Key Features
1. **Side-by-Side Model Comparison**: Compare two models/checkpoints simultaneously
2. **Smart Filtering**:
   - Filter by task names (e.g., g1h1_drawer_rollout, h1g1_make_the_bed)
   - Filter by Text CFG values
   - Filter by Image CFG values
   - Advanced filters for steps and resolution
3. **Optimized Image Display Layout**:
   - **Model 1 (Left)**: Shows source views → edited views
   - **Model 2 (Right)**: Shows edited views → target views
   - Images organized for easy visual comparison
   - Configurable number of examples to display

## Installation

```bash
# Install required packages
pip install streamlit pandas Pillow

# Or use the requirements file
pip install -r requirements_streamlit.txt
```

## Usage

```bash
# Run the application
streamlit run streamlit_compare.py

# The app will be available at http://localhost:8501
```

## Directory Structure
The app expects the following directory structure:
```
/home/liliyu/workspace/BAGEL/results/
├── {run_name}/
│   └── editing_eval/
│       └── {checkpoint_steps}/
│           └── {task}_all_views_raw_renorm{...}/
│               └── {example_id}/
│                   ├── view0_source.png
│                   ├── view0_edited.png
│                   ├── view0_target.png
│                   ├── view1_source.png
│                   ├── view1_edited.png
│                   ├── view1_target.png
│                   └── metadata.jsonl
```

## How to Use

1. **First Launch**: 
   - On first launch, the app will scan the entire results directory
   - This may take a while for large directories
   - Progress will be shown with a progress bar

2. **Select Models**:
   - Choose run_name and checkpoint for Model 1 and Model 2 in the sidebar
   
3. **Apply Filters**:
   - Select tasks you want to compare
   - Choose Text CFG and Image CFG values
   - Use advanced filters for more control

4. **View Results**:
   - Images are displayed side-by-side for both models
   - Each example shows all available views
   - Adjust "Max examples" to control display

5. **Cache Management**:
   - Cache status is shown at the top of the page
   - Use "Force Refresh" to update after new results are generated
   - Use "Clear Cache" to remove all cached data

## Performance Tips

1. **Large Directories**: The caching system is essential for large result directories
2. **First Scan**: Initial scan may take several minutes depending on directory size
3. **Subsequent Loads**: After caching, the app loads almost instantly
4. **Memory Usage**: Limiting displayed examples helps with browser performance

## Troubleshooting

- **No results found**: Check that the results directory exists and contains the expected structure
- **Slow loading**: Use the cached version or limit the number of displayed examples
- **Cache issues**: Use the "Clear Cache" button to reset
- **Images not displaying**: Ensure image files are in PNG format

## Files

- `streamlit_compare.py`: Main application file
- `requirements_streamlit.txt`: Python dependencies
- `.streamlit_cache/results_cache.pkl`: Cached scan results (auto-generated)