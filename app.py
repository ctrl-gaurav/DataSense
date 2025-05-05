import streamlit as st
import pandas as pd
import numpy as np
import utils
import visualizations
import llm_core
import plotly.graph_objects as go
import time
import os
import re # Import re for splitting text
from collections import Counter # For sorting votes

# --- Page Configuration ---
st.set_page_config(
    page_title="DataSense", # Branding
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Configuration ---
# Use session state to load config only once
if 'config' not in st.session_state:
    st.session_state.config = utils.load_config()
    if not st.session_state.config:
        st.error("Failed to load configuration. Please check `config.yaml`.")
        st.stop() # Stop execution if config fails

config = st.session_state.config

# --- Initialize Session State ---
# Use setdefault for cleaner initialization
for key, default_value in {
    "analysis_results": None,
    "current_df": None,
    "current_df_name": None,
    "processing": False,
    "uploaded_filename": None,
    "data_source_idx": 0, # Keep track of radio button index
    "manual_viz_selection": None # For the manual viz explorer
}.items():
    st.session_state.setdefault(key, default_value)


# --- Helper Functions ---
def reset_state():
    # Resets the state when changing data source or uploading new file
    st.session_state.analysis_results = None
    st.session_state.current_df = None
    st.session_state.current_df_name = None
    st.session_state.processing = False
    st.session_state.uploaded_filename = None # Reset uploaded filename tracking
    st.session_state.manual_viz_selection = None # Reset manual selection
    # Clear file uploader state by resetting its key if it exists
    if 'file_uploader_widget' in st.session_state:
         st.session_state.file_uploader_widget = None # Reset key state
    # No automatic rerun here, let the change trigger it naturally or via button

# --- Sidebar ---
st.sidebar.title("📊 DataSense") # Branding
st.sidebar.markdown(
    "Upload your data or select a sample dataset to get automated analysis, "
    "insights, and visualization recommendations from DataSense." # Branding
)

st.sidebar.header("1. Select Data Source")

# Use index to manage radio button state change reliably
data_source_options = ("Upload File", "Sample Dataset")

# Callback function to handle radio button change
def handle_source_change():
     # Only reset if the index actually changed
     new_idx = st.session_state.data_source_selector_idx
     if new_idx != st.session_state.data_source_idx:
          reset_state()
          st.session_state.data_source_idx = new_idx
          # Rerun needed after state reset on source change
          st.rerun()

st.sidebar.radio(
    "Choose data source type:",
    options=range(len(data_source_options)), # Use indices
    format_func=lambda x: data_source_options[x], # Display names
    key="data_source_selector_idx", # Use a different key for the widget itself
    index=st.session_state.data_source_idx,
    on_change=handle_source_change # Use the callback
)
data_source_type = data_source_options[st.session_state.data_source_idx]


uploaded_file = None
selected_dataset = None

# --- Data Loading Logic ---
# This section handles loading based on the selected source type
# It aims to load only when necessary (new file/selection)

if data_source_type == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, JSON, or TXT file",
        type=["csv", "json", "txt"],
        key="file_uploader_widget", # Use a fixed key
        help="Upload your data file (CSV, JSON, or TXT)."
    )
    # Process if a file is uploaded AND it's different from the last processed one
    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_filename:
        reset_state() # Reset everything for the new file
        with st.spinner(f"Loading '{uploaded_file.name}'..."):
            st.session_state.current_df = utils.load_uploaded_file(uploaded_file)
        st.session_state.current_df_name = uploaded_file.name
        st.session_state.uploaded_filename = uploaded_file.name # Track the new file
        if st.session_state.current_df is None:
            st.sidebar.error("Failed to load the uploaded file.")
        else:
            st.sidebar.success(f"Loaded '{uploaded_file.name}'")
        st.rerun() # Rerun to update UI after loading

elif data_source_type == "Sample Dataset":
    sklearn_datasets = utils.get_sklearn_datasets()
    dataset_names = [""] + list(sklearn_datasets.keys()) # Add empty option
    # Get current selection's index, default to 0 if name not found or None
    try:
        # Use get() with default to handle None case gracefully
        current_selection_idx = dataset_names.index(st.session_state.get('current_df_name', None))
    except (ValueError, TypeError): # Handle None or name not in list
        current_selection_idx = 0

    selected_dataset_idx = st.sidebar.selectbox(
        "Select a scikit-learn dataset:",
        options=range(len(dataset_names)),
        format_func=lambda x: dataset_names[x],
        key="dataset_selector_widget",
        index=current_selection_idx,
        help="Choose a sample dataset for analysis."
    )
    selected_dataset_name = dataset_names[selected_dataset_idx]

    # Process if a dataset is selected AND it's different from the current one
    if selected_dataset_name and selected_dataset_name != st.session_state.current_df_name:
        reset_state() # Reset for the new dataset
        with st.spinner(f"Loading '{selected_dataset_name}' dataset..."):
             st.session_state.current_df = utils.load_sklearn_dataset(selected_dataset_name)
        st.session_state.current_df_name = selected_dataset_name
        st.session_state.uploaded_filename = None # Clear uploaded file tracking
        if st.session_state.current_df is None:
            st.sidebar.error(f"Failed to load dataset '{selected_dataset_name}'.")
        else:
            st.sidebar.success(f"Loaded '{selected_dataset_name}' dataset.")
        st.rerun() # Rerun to update UI after loading


# --- Analysis Trigger ---
st.sidebar.header("2. Analyze with DataSense") # Branding
# Disable button if no data OR if processing is ongoing
analyze_button = st.sidebar.button("✨ Analyze and Visualize",
                                 disabled=(st.session_state.current_df is None or st.session_state.processing),
                                 help="Click to start the DataSense analysis.")

st.sidebar.markdown("---")
# Display LLM config in sidebar expander
with st.sidebar.expander("LLM Configuration"):
     # Ensure config is loaded before accessing
     if 'config' in st.session_state and st.session_state.config:
          st.json(st.session_state.config.get('llm_settings', {}))
     else:
          st.warning("Config not loaded.")
st.sidebar.info("Powered by vLLM and Plotly.")


# --- Main Area ---
st.title("📊 DataSense") # Branding
st.markdown("### An Intelligent Data Visualization and Story Generator") # Branding tagline

# Display Preview if data is loaded and we are not processing
if st.session_state.current_df is not None and not st.session_state.processing:
    st.subheader(f"Preview: {st.session_state.current_df_name}")
    st.dataframe(st.session_state.current_df.head(config.get('data_settings', {}).get('max_rows_preview', 5)))
    st.markdown("---") # Separator after preview
# Show info message only if no data is loaded AND we are not processing
elif st.session_state.current_df is None and not st.session_state.processing:
    st.info("Welcome to DataSense! Please select a data source from the sidebar to begin.")


# Handle Analyze Button Click
if analyze_button and st.session_state.current_df is not None:
    st.session_state.processing = True
    st.session_state.analysis_results = None # Clear previous results before starting
    st.session_state.manual_viz_selection = None # Clear manual selection
    st.rerun() # Rerun to show status and start processing


# --- Processing Block ---
# This block now *only* handles the processing and setting the results state
if st.session_state.processing:
    # Use st.status directly for processing feedback and collapsible details
    with st.status("DataSense is analyzing your data...", expanded=True) as status_container:
        analysis_success = False
        # Add progress bar inside the status container
        progress_bar = status_container.progress(0, text="Initializing...")
        try:
            # Ensure config is loaded before initializing processor
            if 'config' not in st.session_state or not st.session_state.config:
                 status_container.update(label="Configuration missing. Cannot proceed.", state="error")
                 raise ValueError("Configuration not loaded.")

            processor = llm_core.DataProcessor(config_path="config.yaml") # Assumes config is loaded

            status_container.update(label=f"Processing dataset: {st.session_state.current_df_name}", state="running")
            progress_bar.progress(0.05, text="Preprocessing & Analyzing Data...") # Initial progress update

            # Ensure current_df is available before processing
            if st.session_state.current_df is None:
                 status_container.update(label="No data loaded to process.", state="error")
                 raise ValueError("Current dataframe is None.")

            # Pass the progress bar object to the processing function
            results = processor.process_dataframe(
                st.session_state.current_df,
                st.session_state.current_df_name,
                progress_bar=progress_bar # Pass progress bar here
                )
            st.session_state.analysis_results = results # Store results regardless of success

            if results.get("success"):
                 progress_bar.progress(1.0, text="Analysis Complete!")
                 status_container.update(label="DataSense analysis complete!", state="complete")
                 analysis_success = True
            else:
                 status_container.update(label="DataSense analysis failed. Check logs/errors.", state="error")
            time.sleep(1) # Keep status visible briefly

        except Exception as e:
            st.error(f"A critical error occurred during analysis: {e}")
            import traceback
            st.error("Traceback:")
            st.code(traceback.format_exc())
            # Store error state in results if not already set
            if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
                 st.session_state.analysis_results = {"success": False, "error": str(e)}
            else: # Ensure success is False if exception happened after results were partially set
                 st.session_state.analysis_results["success"] = False
                 st.session_state.analysis_results["error"] = str(e)

            if 'status_container' in locals():
                 status_container.update(label="Critical error during analysis.", state="error")
        finally:
             st.session_state.processing = False
             # Rerun *after* processing is marked as finished to trigger result display
             st.rerun()


# --- Display Results Block ---
# This block runs *only* when processing is False and results are available
if not st.session_state.processing and st.session_state.analysis_results:
    results = st.session_state.analysis_results

    if results["success"]:
        st.header("📊 DataSense Analysis Results") # Branding

        rec_viz = results["recommendation"]["visualization_type"]
        full_story_text = results["recommendation"]["story"]
        votes = results["recommendation"]["votes"]
        metadata = results["metadata"]
        processed_df = results["processed_dataframe"]
        available_options = results.get("visualization_options", [])

        # --- Display Top Recommendations ---
        st.subheader("🏆 Top Recommendations by DataSense") # Branding

        # Sort votes by count descending, then alphabetically
        sorted_votes = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        top_n = 3 # Show top 3
        displayed_count = 0
        primary_recommendation_displayed = False

        if not sorted_votes:
             st.warning("No specific recommendations generated by the LLM ensemble.")
             # Use Table as default if no votes
             rec_viz = "Table"
             primary_recommendation_displayed = True # Treat Table as displayed
        else:
            # Display top N unique recommendations using columns for layout
            cols = st.columns(min(len(sorted_votes), top_n))
            for i, (viz, count) in enumerate(sorted_votes):
                 if i < top_n:
                     with cols[i]:
                          st.metric(label=f"#{i+1} Recommendation", value=viz, delta=f"{count} Vote(s)")
                          if i == 0:
                               primary_recommendation_displayed = True
                               rec_viz = viz # Ensure rec_viz matches the top voted
                 else:
                      break # Stop after top N

        # Ensure rec_viz is set even if loop didn't run (e.g., only Table available)
        if not primary_recommendation_displayed:
             rec_viz = "Table"

        st.markdown("---")

        # --- Display Primary Visualization & Story ---
        st.subheader(f"Visualizing the Top Recommendation: {rec_viz}")

        if processed_df is None:
             st.error("Processed data is missing, cannot generate visualization or table.")
        elif rec_viz == "Table":
             st.info("The top recommendation is a Table. Displaying the processed data below.")
             st.dataframe(processed_df)
        else:
             # Generate and display Plotly figure for the top recommendation
             with st.spinner(f"Generating {rec_viz}..."):
                 fig = visualizations.create_visualization(processed_df, rec_viz, metadata, config)

             if fig:
                 st.plotly_chart(fig, use_container_width=True)
             else: # Handle cases where viz generation failed
                 st.warning(f"Could not generate the recommended '{rec_viz}'. Displaying data table instead.")
                 if processed_df is not None:
                      st.dataframe(processed_df)
                 else:
                      st.error("Processed data is missing, cannot display table fallback.")

        # --- Display Insights and Story for the Primary Recommendation ---
        st.markdown("---") # Add a separator
        st.subheader(f"DataSense Analysis & Story for {rec_viz}") # Branding

        # Try to split the story based on headings
        insights_section = None
        story_section = None
        other_details = "" # Store text not under specific headings

        if isinstance(full_story_text, str): # Ensure it's a string before regex
            insights_match = re.search(r'^###\s+Data Insights\s*$(.*?)($|(?=^###\s+))', full_story_text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            story_match = re.search(r'^###\s+Data Story\s*$(.*?)(?=$)', full_story_text, re.MULTILINE | re.IGNORECASE | re.DOTALL)

            if insights_match: insights_section = insights_match.group(1).strip()
            if story_match: story_section = story_match.group(1).strip()

            first_heading_pos = len(full_story_text)
            if insights_match: first_heading_pos = min(first_heading_pos, insights_match.start())
            if story_match: first_heading_pos = min(first_heading_pos, story_match.start())
            other_details = full_story_text[:first_heading_pos].strip()

            if other_details:
                 st.markdown("#### Reasoning / Overview")
                 st.markdown(other_details)
            if insights_section:
                 st.markdown("#### Data Insights")
                 st.markdown(insights_section)
            if story_section:
                 st.markdown("#### Data Story")
                 st.markdown(story_section)
            if not insights_section and not story_section and not other_details and full_story_text:
                 st.markdown("#### Analysis Text")
                 st.markdown(full_story_text)
        else:
             st.warning("Analysis text is not available or not in the expected format.")

        # --- Manual Visualization Explorer ---
        st.markdown("---")
        st.subheader("🔬 Manual Visualization Explorer")
        if not available_options:
             st.warning("No visualization options were generated based on the data analysis.")
        elif processed_df is None:
             st.error("Processed data is missing, cannot generate manual visualizations.")
        else:
            # Use session state to preserve selection across reruns if needed
            st.session_state.manual_viz_selection = st.selectbox(
                 "Select a visualization type to generate manually:",
                 options=[""] + available_options, # Add empty default option
                 key="manual_viz_selector",
                 index=0, # Default to empty
                 help="Choose any available visualization type to see how it looks with your data."
            )

            selected_manual_viz = st.session_state.manual_viz_selection
            if selected_manual_viz:
                 st.markdown(f"#### Preview: {selected_manual_viz}")
                 with st.spinner(f"Generating {selected_manual_viz}..."):
                      manual_fig = visualizations.create_visualization(processed_df, selected_manual_viz, metadata, config)

                 if manual_fig:
                      st.plotly_chart(manual_fig, use_container_width=True)
                 else:
                      st.warning(f"Could not generate '{selected_manual_viz}'. It might not be suitable or implemented.")


        # --- Expanders for Details ---
        st.markdown("---") # Separator

        # These expanders are now rendered when processing is False, avoiding nesting issues
        with st.expander("LLM Voting Details (Full)"):
             st.write("Votes for each visualization type across all model responses:")
             st.json(votes) # Show all votes here

        with st.expander("Dataset Metadata Summary"):
             if metadata:
                  st.write(f"- **Rows:** {metadata.get('num_rows', 'N/A')}")
                  st.write(f"- **Columns:** {metadata.get('num_columns', 'N/A')}")
                  st.write(f"- **Numeric Cols:** {len(metadata.get('potential_numeric',[]))}")
                  st.write(f"- **Categorical/Bool Cols:** {len(metadata.get('potential_categorical',[])) + len(metadata.get('potential_boolean',[]))}")
                  st.write(f"- **Datetime Cols:** {len(metadata.get('potential_datetime',[]))}")
                  st.write(f"- **Text Cols:** {len(metadata.get('potential_text',[]))}")
                  if metadata.get("missing_values"):
                       st.write("- **Cols with Missing Values:**")
                       missing_vals = metadata["missing_values"]
                       # Show limited number if many columns have missing vals
                       if len(missing_vals) > 10:
                            st.json(dict(list(missing_vals.items())[:10]))
                            st.write(f"...and {len(missing_vals)-10} more.")
                       else:
                            st.json(missing_vals)
             else:
                  st.write("Metadata not available.")

        with st.expander("Available Visualization Options Considered by DataSense"): # Branding
             st.write("Based on data analysis, the following visualizations were considered:")
             st.write(available_options)

    # Handle Analysis Failure Case
    elif not st.session_state.processing: # Ensure processing is finished before showing failure
        st.error("DataSense Analysis Failed") # Branding
        st.error(f"Error details: {results.get('error', 'Unknown error')}")

