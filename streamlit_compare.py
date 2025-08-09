import re
import streamlit as st
import pandas as pd
from pathlib import Path

import json
import pickle
import time
import base64

from datetime import datetime, timedelta

# Set page config for wide layout
st.set_page_config(
    page_title="BAGEL Image Generation Comparison Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stImage {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        margin: 5px;
    }
    .view-label {
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .example-header {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache file path
CACHE_FILE = Path("/home/liliyu/workspace/BAGEL/.streamlit_cache/results_cache.pkl")
CACHE_FILE.parent.mkdir(exist_ok=True)

def load_cached_data():
    """Load cached data from disk if available and not too old"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                # Check if cache is less than 24 hours old
                if datetime.now() - cache_data['timestamp'] < timedelta(hours=24):
                    return cache_data['data']
        except Exception as e:
            st.warning(f"Could not load cache: {e}")
    return None

def save_cache(data):
    """Save data to cache file"""
    try:
        cache_data = {
            'timestamp': datetime.now(),
            'data': data
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        st.warning(f"Could not save cache: {e}")

def scan_results_directory_impl(base_path="/home/liliyu/workspace/BAGEL/results", progress_callback=None, count_examples=False):
    """Implementation of directory scanning with progress callback"""
    data = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        return pd.DataFrame()
    
    # Pattern for parsing hyperparameter folders
    hyperparam_pattern = re.compile(
        r"(?P<task>.+?)_all_views_raw_renorm(?P<renorm>[\d.]+)_text(?P<textcfg>[\d.]+)_img(?P<imgcfg>[\d.]+)_shift(?P<shift>[\d.]+)_nsteps(?P<nsteps>\d+)_res(?P<res>\d+)_heun(?P<heun>\w+)_condition(?P<condition>\w+)_?"
    )
    
    # Filter directories to scan - only relevant ones
    # More specific filtering to speed up scanning
    run_dirs = []
    for d in base_path.iterdir():
        if not d.is_dir():
            continue
        _name = d.name
        # Only include directories that match our patterns
        # if (name.startswith("seed_blip3o_all_robots") or 
        #     name.startswith("h1g1_vit") or
        #     name.startswith("seed_unddata") or
        #     name.startswith("seed_all_robots") or
        #     name.startswith("TEST_seed")):
        # if "h1g1" in name:
        # Check if it has editing_eval subdirectory
        if (d / "editing_eval").exists():
            run_dirs.append(d)
    
    total_runs = len(run_dirs)
    
    if total_runs == 0:
        if progress_callback:
            progress_callback(1.0, "No matching directories found")
        return pd.DataFrame()
    
    # Scan all directories
    for idx, run_dir in enumerate(run_dirs):
        # Update progress if callback provided
        if progress_callback:
            progress_callback((idx + 1) / total_runs, f"Scanning: {run_dir.name} ({idx+1}/{total_runs})")
        
        editing_eval_path = run_dir / "editing_eval"
        run_name = run_dir.name
        
        # Scan checkpoints
        for checkpoint_dir in editing_eval_path.iterdir():
            if not checkpoint_dir.is_dir():
                continue
                
            checkpoint_steps = checkpoint_dir.name
            
            # Scan hyperparameter directories
            for hyperparam_dir in checkpoint_dir.iterdir():
                if not hyperparam_dir.is_dir():
                    continue
                    
                match = hyperparam_pattern.match(hyperparam_dir.name)
                if match:
                    info = match.groupdict()
                    info['run_name'] = run_name
                    info['checkpoint_steps'] = checkpoint_steps
                    info['hyperparam_dir'] = hyperparam_dir.name
                    info['full_path'] = str(hyperparam_dir)
                    
                    # Only count examples if requested (slower)
                    if count_examples:
                        try:
                            example_count = sum(1 for item in hyperparam_dir.iterdir() if item.is_dir())
                        except Exception:
                            example_count = 0
                    else:
                        example_count = -1  # Indicate not counted
                    info['example_count'] = example_count
                    
                    data.append(info)
    
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)  # Cache for 1 hour in memory
def scan_results_directory(base_path="/home/liliyu/workspace/BAGEL/results", force_refresh=False):
    """Scan the results directory to find all available models and checkpoints"""
    
    # Try to load from cache first if not forcing refresh
    if not force_refresh:
        cached_data = load_cached_data()
        if cached_data is not None:
            return cached_data
    
    # If we need to scan, use the implementation function
    # This allows us to show progress properly
    df = scan_results_directory_impl(base_path)
    
    # Save to cache
    if not df.empty:
        save_cache(df)
    
    return df

# Helper to display images with optional highlight background

def display_image_with_bg(b64_str: str, highlight: bool = False):
    """Render a base64-encoded PNG with optional pink background."""
    style_bg = "background-color:#ffe6f2;padding:10px;border-radius:8px;" if highlight else ""
    st.markdown(
        f"<div style='{style_bg}'>"
        f"<img src='data:image/png;base64,{b64_str}' style='width:100%;object-fit:contain;'/></div>",
        unsafe_allow_html=True,
    )


@st.cache_data
def load_images_for_example(example_path):
    """Load all images for a specific example"""
    images = {}
    example_path = Path(example_path)
    
    if not example_path.exists():
        return images
    
    # Load and encode image files once (base64)
    for img_file in example_path.glob("*.png"):
        # Parse filename to extract view and type
        filename = img_file.stem
        if "view" in filename:
            parts = filename.split("_")
            view = parts[0]
            img_type = "_".join(parts[1:])
            try:
                with open(img_file, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                images[f"{view}_{img_type}"] = b64
            except Exception as e:
                st.error(f"Error loading image {img_file}: {e}")
    
    # Also try to load metadata if available
    metadata = None
    # Try .jsonl first
    metadata_file = example_path / "metadata.jsonl"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                first_line = f.readline()
                metadata = json.loads(first_line) if first_line else None
        except Exception:
            metadata = None
    # Fallback to metadata.json
    if metadata is None:
        metadata_file = example_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = None
    
    return images, metadata

def display_model_results(model_data, column, model_num):
    """Display results for a single model in a column"""
    with column:
        st.markdown(f"### 📊 Model {model_num}")
        
        if model_data.empty:
            st.warning("No data found for selected filters")
            return
        
        # Display model info
        st.info(f"**Run:** {model_data.iloc[0]['run_name']}\n**Checkpoint:** {model_data.iloc[0]['checkpoint_steps']}")
        
        # Limit number of examples to display (once per model, not per task)
        max_examples = st.sidebar.number_input(f"Max examples for Model {model_num}", min_value=1, max_value=50, value=20, key=f"max_ex_{model_num}")
        
        # Group by task
        tasks = model_data['task'].unique()
        
        for task in sorted(tasks):
            task_data = model_data[model_data['task'] == task]
            
            st.markdown(f"#### Task: {task}")
            
            # Display hyperparameters for this task/configuration
            first_row = task_data.iloc[0]
            textcfg_val = first_row.get('textcfg', first_row.get('cfg', 'N/A'))
            imgcfg_val = first_row.get('imgcfg', 'N/A')
            nsteps_val = first_row.get('nsteps', first_row.get('steps', 'N/A'))
            res_val = first_row.get('res', 'N/A')
            st.caption(
                f"TextCFG: {textcfg_val} | ImgCFG: {imgcfg_val} | "
                f"Steps: {nsteps_val} | Resolution: {res_val}"
            )
            
            # Get example directories
            example_dir = Path(first_row['full_path'])
            example_dirs = sorted([d for d in example_dir.iterdir() if d.is_dir()])
            
            for example_idx, example_path in enumerate(example_dirs[:max_examples]):
                images, metadata = load_images_for_example(example_path)
                
                if not images:
                    continue
                
                st.markdown(f"**Example {example_path.name}**")
                
                # Show instruction text if available; show placeholder for alignment when missing
                if metadata and isinstance(metadata, dict) and 'instruction' in metadata:
                    st.markdown(
                        f"<div style='font-size:20px;font-weight:600;margin:6px 0;color:#2ecc71;'>📝 Instruction: {metadata['instruction']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
                
                # Organize images by view
                views = set([key.split('_')[0] for key in images.keys()])
                views = sorted(views)
                
                # Different display order based on model number
                if model_num == 1:
                    # Model 1 (left): Show source views, then edited views
                    # First show all source views
                    st.markdown("**Source Views:**")
                    source_cols = st.columns(len(views))
                    for i, view in enumerate(views):
                        key = f"{view}_source"
                        if key in images:
                            with source_cols[i]:
                                st.caption(f"source_{view}")
                                display_image_with_bg(images[key], highlight=False)
                    
                    # Then show all edited views
                    st.markdown("**Edited Views:**")
                    edited_cols = st.columns(len(views))
                    for i, view in enumerate(views):
                        key = f"{view}_edited"
                        if key in images:
                            with edited_cols[i]:
                                st.caption(f"edited_{view}")
                                display_image_with_bg(images[key], highlight=True)
                
                else:  # model_num == 2
                    # Model 2 (right): Show target views first, then edited views
                    st.markdown("**Target Views:**")
                    target_cols = st.columns(len(views))
                    for i, view in enumerate(views):
                        key = f"{view}_target"
                        if key in images:
                            with target_cols[i]:
                                st.caption(f"target_{view}")
                                display_image_with_bg(images[key], highlight=False)
                    
                    # Then show all edited views
                    st.markdown("**Edited Views:**")
                    edited_cols = st.columns(len(views))
                    for i, view in enumerate(views):
                        key = f"{view}_edited"
                        if key in images:
                            with edited_cols[i]:
                                st.caption(f"edited_{view}")
                                display_image_with_bg(images[key], highlight=True)
                
                st.divider()

def main():
    st.title("🎨 BAGEL Image Generation Comparison Tool")
    st.markdown("Compare generated images from different models and checkpoints side-by-side")
    
    # Initialize session state for cache management
    if 'force_refresh' not in st.session_state:
        st.session_state.force_refresh = False
    
    # Cache management UI
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        # Check if cache exists and show its age
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                    cache_age = datetime.now() - cache_data['timestamp']
                    hours_old = cache_age.total_seconds() / 3600
                    if hours_old < 1:
                        st.success(f"✅ Using cached data (updated {int(cache_age.total_seconds() / 60)} minutes ago)")
                    elif hours_old < 24:
                        st.info(f"📊 Using cached data (updated {hours_old:.1f} hours ago)")
                    else:
                        st.warning(f"⚠️ Cache is old ({hours_old:.1f} hours). Consider refreshing.")
            except Exception:
                st.info("📊 Cache available")
        else:
            st.info("🔍 No cache found. Will scan directory...")
    
    with col2:
        if st.button("🔄 Force Refresh", help="Re-scan the entire results directory"):
            st.session_state.force_refresh = True
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Cache", help="Delete cached data"):
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
                st.cache_data.clear()
                st.rerun()
    
    # Scan available data
    if st.session_state.force_refresh:
        st.markdown("### 🔍 Scanning Directory Structure...")
        
        # Show progress while scanning
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            info_text = st.empty()
        
        # Define progress callback
        def update_progress(progress, status):
            progress_bar.progress(progress)
            status_text.text(status)
            info_text.caption("💡 Tip: This scan only happens once. Results are cached for future use.")
        
        # Scan with progress updates (without counting examples for speed)
        df = scan_results_directory_impl(
            base_path="/mnt/weka/checkpoints/liliyu/bagel_ckpt",
            progress_callback=update_progress,
            count_examples=False  # Skip counting for speed
        )
        
        # Save to cache
        if not df.empty:
            save_cache(df)
            progress_container.success(f"✅ Scan complete! Found {len(df)} configurations.")
        else:
            progress_container.warning("⚠️ No matching directories found.")
        
        # Clear progress indicators after a short delay
        time.sleep(1)
        progress_container.empty()
        
        # Reset force_refresh flag
        st.session_state.force_refresh = False
        
        # Clear the in-memory cache to use the new data
        st.cache_data.clear()
    else:
        df = scan_results_directory(force_refresh=False)
    
    if df.empty:
        st.error("No results found in the directory. Please check the path.")
        return
    
    # Display statistics
    st.markdown(f"**📈 Found:** {len(df)} configurations across {df['run_name'].nunique()} models")

    # Model Selection (main panel)
    st.markdown("### 🔧 Model Selection")
    model_col1, model_col2 = st.columns(2)
    run_names = sorted(df['run_name'].unique())
    with model_col1:
        st.markdown("**Model 1**")
        run_name_1 = st.selectbox("Run Name - Model 1", run_names, key="run1_main")
        checkpoints_1 = sorted(df[df['run_name'] == run_name_1]['checkpoint_steps'].unique())
        checkpoint_1 = st.selectbox("Checkpoint - Model 1", checkpoints_1, key="ckpt1_main")
    with model_col2:
        st.markdown("**Model 2**")
        default_idx = min(1, len(run_names) - 1)
        run_name_2 = st.selectbox("Run Name - Model 2", run_names, index=default_idx, key="run2_main")
        checkpoints_2 = sorted(df[df['run_name'] == run_name_2]['checkpoint_steps'].unique())
        checkpoint_2 = st.selectbox("Checkpoint - Model 2", checkpoints_2, key="ckpt2_main")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Filters")
        

        
        st.divider()
        
        # Task and hyperparameter filters
        st.subheader("Task & Hyperparameter Filters")
        
        # Get available options based on selected models
        model1_data = df[(df['run_name'] == run_name_1) & (df['checkpoint_steps'] == checkpoint_1)]
        model2_data = df[(df['run_name'] == run_name_2) & (df['checkpoint_steps'] == checkpoint_2)]
        combined_data = pd.concat([model1_data, model2_data])
        
        # Task filter
        available_tasks = sorted(combined_data['task'].unique()) if 'task' in combined_data.columns else []
        selected_tasks = st.multiselect("Tasks", available_tasks, default=available_tasks[:1] if available_tasks else [])
        
        # TextCFG filter
        if 'textcfg' in combined_data.columns:
            available_textcfg = sorted(combined_data['textcfg'].dropna().unique())
        else:
            available_textcfg = []
        selected_textcfg = st.multiselect("Text CFG", available_textcfg, default=available_textcfg[:1] if available_textcfg else [])
        
        # ImgCFG filter
        if 'imgcfg' in combined_data.columns:
            available_imgcfg = sorted(combined_data['imgcfg'].dropna().unique())
        else:
            available_imgcfg = []
        selected_imgcfg = st.multiselect("Image CFG", available_imgcfg, default=available_imgcfg[:1] if available_imgcfg else [])
        
        # Additional filters
        with st.expander("Advanced Filters"):
            if 'nsteps' in combined_data.columns:
                available_nsteps = sorted(combined_data['nsteps'].dropna().unique())
            else:
                available_nsteps = []
            selected_nsteps = st.multiselect("Number of Steps", available_nsteps, default=available_nsteps)
            
            if 'res' in combined_data.columns:
                available_res = sorted(combined_data['res'].dropna().unique())
            else:
                available_res = []
            selected_res = st.multiselect("Resolution", available_res, default=available_res)
    
    # Apply filters
    if selected_tasks:
        # Helper to apply filters safely
        def apply_filters(df_src):
            df_out = df_src[df_src['task'].isin(selected_tasks)] if 'task' in df_src.columns else df_src
            if selected_textcfg and 'textcfg' in df_out.columns:
                df_out = df_out[df_out['textcfg'].isin(selected_textcfg)]
            if selected_imgcfg and 'imgcfg' in df_out.columns:
                df_out = df_out[df_out['imgcfg'].isin(selected_imgcfg)]
            if selected_nsteps and 'nsteps' in df_out.columns:
                df_out = df_out[df_out['nsteps'].isin(selected_nsteps)]
            if selected_res and 'res' in df_out.columns:
                df_out = df_out[df_out['res'].isin(selected_res)]
            return df_out
        
        model1_filtered = apply_filters(model1_data)
        model2_filtered = apply_filters(model2_data)
        
        # Display results side by side
        col1, col2 = st.columns(2)
        
        display_model_results(model1_filtered, col1, 1)
        display_model_results(model2_filtered, col2, 2)
    else:
        st.warning("Please select at least one option for Tasks to display results.")
    
    # Footer
    st.divider()
    st.caption("BAGEL Image Generation Comparison Tool | Generated images from eval/gen/gen_images_edit_allviews_ddp.py")

if __name__ == "__main__":
    main()