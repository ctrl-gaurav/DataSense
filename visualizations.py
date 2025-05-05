import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import streamlit as st

# Define a base template for consistent styling
plotly_template = "plotly_white" # Use a clean base theme

def handle_missing_values(df: pd.DataFrame, columns: List[str], strategy: str = 'drop') -> pd.DataFrame:
    """Handles missing values for plotting."""
    df_copy = df.copy()
    if not columns: # If no specific columns provided, maybe operate on all? Or return as is?
        return df_copy # Return copy if no columns specified for handling

    if strategy == 'drop':
        # Drop rows where *any* of the specified columns are NaN
        return df_copy.dropna(subset=columns)
    elif strategy == 'impute_mean': # Example imputation (use cautiously, only for numeric)
        for col in columns:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                mean_val = df_copy[col].mean()
                if pd.notna(mean_val): # Check if mean is valid
                     df_copy[col].fillna(mean_val, inplace=True)
                else: # Handle case where mean is NaN (e.g., all NaNs in column)
                     df_copy[col].fillna(0, inplace=True) # Or some other default
        return df_copy
    # Add other strategies if needed (e.g., median, mode)
    return df_copy

def create_visualization(df: pd.DataFrame, viz_type: str, metadata: Dict[str, Any], config: Dict[str, Any]) -> Optional[go.Figure]:
    """Generates the specified Plotly visualization with enhanced aesthetics."""
    if df is None or df.empty:
        st.warning(f"Cannot generate '{viz_type}': Input data is empty.")
        return None

    fig = None
    numeric_cols = metadata.get("potential_numeric", [])
    categorical_cols = metadata.get("potential_categorical", [])
    datetime_cols = metadata.get("potential_datetime", [])
    boolean_cols = metadata.get("potential_boolean", [])
    all_categoricals = categorical_cols + boolean_cols # Treat booleans like categoricals for plotting

    # Ensure columns actually exist in the dataframe provided
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    all_categoricals = [col for col in all_categoricals if col in df.columns]
    datetime_cols = [col for col in datetime_cols if col in df.columns]

    try:
        # Apply base template to all plots created with px
        if viz_type == "Histogram" and numeric_cols:
            col = numeric_cols[0]
            plot_df = handle_missing_values(df, [col], strategy='drop')
            if not plot_df.empty:
                 fig = px.histogram(plot_df, x=col, title=f"Distribution of {col}",
                                    marginal="box", # Keep box for distribution summary
                                    opacity=0.75, # Slightly transparent bars
                                    template=plotly_template)
                 fig.update_layout(bargap=0.1) # Add gap between bars
            else: st.warning(f"No data left for Histogram of '{col}' after handling missing values.")

        elif viz_type == "Box Plot" and numeric_cols:
            col = numeric_cols[0]
            plot_df = handle_missing_values(df, [col], strategy='drop')
            if not plot_df.empty:
                 # Use graph_objects for more control if needed, or stick with px
                 fig = px.box(plot_df, y=col, title=f"Box Plot of {col}",
                              points="outliers", # Show outliers clearly
                              template=plotly_template)
                 # Enhance appearance
                 fig.update_traces(marker=dict(size=3, opacity=0.7))
            else: st.warning(f"No data left for Box Plot of '{col}' after handling missing values.")

        elif viz_type == "Density Plot" and numeric_cols:
            col = numeric_cols[0]
            plot_df = handle_missing_values(df, [col], strategy='drop')
            if not plot_df.empty:
                try:
                    import plotly.figure_factory as ff
                    hist_data = [plot_df[col].values]
                    group_labels = [col]
                    # Create the distplot
                    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2,
                                             show_hist=False, show_rug=False)
                    # Apply template styling after creation for ff plots
                    fig.update_layout(title=f"Density Plot of {col}", template=plotly_template)
                    # Enhance line appearance
                    fig.update_traces(line=dict(width=2.5))
                except ImportError:
                     st.warning("Cannot create Density Plot: `scipy` library not installed. Showing Histogram instead.")
                     fig = px.histogram(plot_df, x=col, title=f"Distribution of {col} (Histogram fallback)", opacity=0.75, template=plotly_template)
                     fig.update_layout(bargap=0.1)
                except Exception as e:
                     st.error(f"Error creating Density Plot for {col}: {e}. Showing Histogram.")
                     fig = px.histogram(plot_df, x=col, title=f"Distribution of {col} (Histogram fallback)", opacity=0.75, template=plotly_template)
                     fig.update_layout(bargap=0.1)
            else: st.warning(f"No data left for Density Plot of '{col}' after handling missing values.")


        elif viz_type == "Bar Chart (Counts)" and all_categoricals:
            col = all_categoricals[0]
            plot_df = handle_missing_values(df, [col], strategy='drop')
            if not plot_df.empty:
                counts = plot_df[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                if len(counts) > 50:
                     counts = counts.nlargest(50, 'count')
                     st.warning(f"Showing top 50 categories for '{col}' in Bar Chart.")
                fig = px.bar(counts, x=col, y='count', title=f"Counts of {col}",
                             template=plotly_template, text_auto=True, # Add counts on bars
                             color_discrete_sequence=px.colors.qualitative.Pastel) # Softer colors
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis={'categoryorder':'total descending'}, bargap=0.15)
            else: st.warning(f"No data left for Bar Chart of '{col}' after handling missing values.")

        elif viz_type == "Table":
            # Handled directly in app.py
            return None

        else:
            st.warning(f"Visualization type '{viz_type}' is not currently implemented or not suitable.")


        # --- Final Common Layout Updates ---
        if fig:
            fig.update_layout(
                title_x=0.5, # Center title
                font=dict(family="Arial, sans-serif", size=12, color="#333"), # Darker font
                # Keep transparent background for embedding flexibility
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend_title_text=fig.layout.legend.title.text if fig.layout.legend and fig.layout.legend.title and fig.layout.legend.title.text else 'Legend', # Keep legend title if exists
                margin=dict(l=50, r=30, t=70, b=50) # Adjust margins
            )
            # Add subtle grid lines if using plotly_white template
            if plotly_template == "plotly_white":
                 fig.update_layout(xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.5)'),
                                   yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.5)'))
            # Ensure responsiveness
            fig.update_layout(autosize=True)


    except Exception as e:
        st.error(f"Failed to generate visualization '{viz_type}': {e}")
        import traceback
        st.code(traceback.format_exc()) # Show traceback for debugging
        fig = None # Return None on error

    # If no figure was generated for any reason (empty data, unimplemented type, error)
    if fig is None and viz_type != "Table":
         # Create a placeholder figure indicating the issue
         fig = go.Figure()
         fig.update_layout(
              title=f"{viz_type} (Not Available or Error)",
              title_x=0.5,
              xaxis={"visible": False},
              yaxis={"visible": False},
              paper_bgcolor='rgba(0,0,0,0)',
              plot_bgcolor='rgba(0,0,0,0)',
              annotations=[{
                    "text": f"Could not generate '{viz_type}'.<br>Check warnings/errors above.",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14, "color": "#888"}
              }]
         )

    return fig
