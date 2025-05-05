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

        elif viz_type == "Pie Chart" and all_categoricals:
            col = all_categoricals[0]
            plot_df = handle_missing_values(df, [col], strategy='drop')
            if not plot_df.empty:
                counts = plot_df[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                if len(counts) > 15:
                     counts = counts.nlargest(15, 'count')
                     st.warning(f"Showing top 15 categories for '{col}' in Pie Chart.")
                fig = px.pie(counts, names=col, values='count', title=f"Proportion of {col}",
                             template=plotly_template, hole=0.3) # Add a donut hole
                fig.update_traces(textinfo='percent+label', pull=[0.05] * len(counts)) # Pull slices slightly
            else: st.warning(f"No data left for Pie Chart of '{col}' after handling missing values.")


        elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            plot_df = handle_missing_values(df, [x_col, y_col], strategy='drop')
            if not plot_df.empty:
                 fig = px.scatter(plot_df, x=x_col, y=y_col, title=f"Scatter Plot: {y_col} vs {x_col}",
                                  trendline="ols", # Ordinary Least Squares trendline
                                  trendline_color_override="rgba(255,0,0,0.6)", # Red trendline
                                  opacity=0.6, # Make points slightly transparent
                                  template=plotly_template)
                 # Enhance marker style
                 fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            else: st.warning(f"No data left for Scatter Plot of '{y_col}' vs '{x_col}' after handling missing values.")

        elif viz_type == "Heatmap (Correlation)" and len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            if not corr.empty:
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                                title="Numeric Feature Correlation Heatmap",
                                color_continuous_scale='Viridis', # Perceptually uniform colorscale
                                zmin=-1, zmax=1, template=plotly_template)
            else:
                 st.warning("Not enough numeric data or variance to create correlation heatmap.")

        elif viz_type == "Box Plot by Category" and numeric_cols and all_categoricals:
            num_col = numeric_cols[0]
            cat_col = all_categoricals[0]
            plot_df = handle_missing_values(df, [num_col, cat_col], strategy='drop')
            if not plot_df.empty:
                unique_cats = plot_df[cat_col].nunique()
                if unique_cats > 20:
                     top_cats = plot_df[cat_col].value_counts().nlargest(20).index
                     plot_df = plot_df[plot_df[cat_col].isin(top_cats)]
                     st.warning(f"Showing Box Plot for top 20 categories of '{cat_col}'.")
                fig = px.box(plot_df, x=cat_col, y=num_col, title=f"Box Plot of {num_col} by {cat_col}",
                             points="outliers", template=plotly_template,
                             color=cat_col, color_discrete_sequence=px.colors.qualitative.Pastel) # Color boxes
            else: st.warning(f"No data left for Box Plot of '{num_col}' by '{cat_col}' after handling missing values.")

        elif viz_type == "Violin Plot by Category" and numeric_cols and all_categoricals:
            num_col = numeric_cols[0]
            cat_col = all_categoricals[0]
            plot_df = handle_missing_values(df, [num_col, cat_col], strategy='drop')
            if not plot_df.empty:
                unique_cats = plot_df[cat_col].nunique()
                if unique_cats > 20:
                     top_cats = plot_df[cat_col].value_counts().nlargest(20).index
                     plot_df = plot_df[plot_df[cat_col].isin(top_cats)]
                     st.warning(f"Showing Violin Plot for top 20 categories of '{cat_col}'.")
                fig = px.violin(plot_df, x=cat_col, y=num_col, title=f"Violin Plot of {num_col} by {cat_col}",
                                box=True, points="outliers", template=plotly_template,
                                color=cat_col, color_discrete_sequence=px.colors.qualitative.Pastel) # Color violins
            else: st.warning(f"No data left for Violin Plot of '{num_col}' by '{cat_col}' after handling missing values.")


        elif viz_type == "Bar Chart (Aggregated)" and numeric_cols and all_categoricals:
            num_col = numeric_cols[0]
            cat_col = all_categoricals[0]
            plot_df = handle_missing_values(df, [num_col, cat_col], strategy='drop')
            if not plot_df.empty:
                agg_df = plot_df.groupby(cat_col, observed=False)[num_col].mean().reset_index()
                if len(agg_df) > 50:
                     agg_df = agg_df.nlargest(50, num_col)
                     st.warning(f"Showing top 50 categories for '{cat_col}' in Aggregated Bar Chart (mean).")
                fig = px.bar(agg_df, x=cat_col, y=num_col, title=f"Mean {num_col} by {cat_col}",
                             template=plotly_template, text_auto=True,
                             color=cat_col, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis={'categoryorder':'total descending'}, bargap=0.15, showlegend=False) # Hide legend if coloring by x-axis
            else: st.warning(f"No data left for Aggregated Bar Chart of '{num_col}' by '{cat_col}' after handling missing values.")


        elif viz_type == "Line Chart" and numeric_cols and datetime_cols:
            num_col = numeric_cols[0]
            date_col = datetime_cols[0]
            plot_df = handle_missing_values(df, [num_col, date_col], strategy='drop').sort_values(by=date_col)
            if not plot_df.empty:
                 fig = px.line(plot_df, x=date_col, y=num_col, title=f"Line Chart: {num_col} over Time ({date_col})",
                               markers=True, template=plotly_template)
                 fig.update_traces(line=dict(width=2.5))
            else: st.warning(f"No data left for Line Chart of '{num_col}' vs '{date_col}' after handling missing values.")

        elif viz_type == "Area Chart" and numeric_cols and datetime_cols:
            num_col = numeric_cols[0]
            date_col = datetime_cols[0]
            plot_df = handle_missing_values(df, [num_col, date_col], strategy='drop').sort_values(by=date_col)
            if not plot_df.empty:
                 fig = px.area(plot_df, x=date_col, y=num_col, title=f"Area Chart: {num_col} over Time ({date_col})",
                               template=plotly_template)
                 fig.update_traces(line=dict(width=0.5)) # Thinner line for area
            else: st.warning(f"No data left for Area Chart of '{num_col}' vs '{date_col}' after handling missing values.")


        elif viz_type == "Stacked Bar Chart" and len(all_categoricals) >= 2:
             cat_col1 = all_categoricals[0]
             cat_col2 = all_categoricals[1]
             plot_df = handle_missing_values(df, [cat_col1, cat_col2], strategy='drop')
             if not plot_df.empty:
                 unique_cats1 = plot_df[cat_col1].nunique()
                 unique_cats2 = plot_df[cat_col2].nunique()
                 if unique_cats1 > 50 or unique_cats2 > 20:
                     st.warning(f"Too many categories for Stacked Bar Chart ({unique_cats1}x{unique_cats2}).")
                 else:
                     counts = plot_df.groupby([cat_col1, cat_col2], observed=False).size().reset_index(name='count')
                     fig = px.bar(counts, x=cat_col1, y='count', color=cat_col2,
                                  title=f"Stacked Bar Chart: Counts by {cat_col1} and {cat_col2}",
                                  template=plotly_template)
                     fig.update_layout(bargap=0.15)
             else: st.warning(f"No data left for Stacked Bar Chart of '{cat_col1}' vs '{cat_col2}' after handling missing values.")


        elif viz_type == "Grouped Bar Chart" and len(all_categoricals) >= 2:
             cat_col1 = all_categoricals[0]
             cat_col2 = all_categoricals[1]
             plot_df = handle_missing_values(df, [cat_col1, cat_col2], strategy='drop')
             if not plot_df.empty:
                 unique_cats1 = plot_df[cat_col1].nunique()
                 unique_cats2 = plot_df[cat_col2].nunique()
                 if unique_cats1 > 50 or unique_cats2 > 20:
                     st.warning(f"Too many categories for Grouped Bar Chart ({unique_cats1}x{unique_cats2}).")
                 else:
                     counts = plot_df.groupby([cat_col1, cat_col2], observed=False).size().reset_index(name='count')
                     fig = px.bar(counts, x=cat_col1, y='count', color=cat_col2, barmode='group',
                                  title=f"Grouped Bar Chart: Counts by {cat_col1} and {cat_col2}",
                                  template=plotly_template, text_auto=True)
                     fig.update_traces(textposition='outside')
                     fig.update_layout(bargap=0.15)
             else: st.warning(f"No data left for Grouped Bar Chart of '{cat_col1}' vs '{cat_col2}' after handling missing values.")

        
        elif viz_type == "3D Scatter Plot" and len(numeric_cols) >= 3:
            x_col, y_col, z_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
            color_col = all_categoricals[0] if all_categoricals else None
            cols_to_check = [x_col, y_col, z_col] + ([color_col] if color_col else [])
            plot_df = handle_missing_values(df, cols_to_check, strategy='drop')
            if not plot_df.empty:
                 # Limit color categories if too many
                 final_color_col = None
                 if color_col:
                      unique_colors = plot_df[color_col].nunique()
                      if unique_colors <= 20:
                          final_color_col = color_col
                      else:
                          st.warning(f"Too many categories ({unique_colors}) in '{color_col}' for 3D Scatter color. Plotting without color.")

                 fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=final_color_col,
                                     title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}" + (f" colored by {final_color_col}" if final_color_col else ""))
            else: st.warning(f"No data left for 3D Scatter Plot after handling missing values.")





        # elif viz_type == "3D Scatter Plot" and len(numeric_cols) >= 3:
        #     x_col, y_col, z_col = numeric_cols[0], numeric_cols[1], numeric_cols[2]
        #     color_col = all_categoricals[0] if all_categoricals else None
        #     cols_to_check = [x_col, y_col, z_col] + ([color_col] if color_col else [])
        #     plot_df = handle_missing_values(df, cols_to_check, strategy='drop')
        #     if not plot_df.empty:
        #          final_color_col = None
        #          if color_col:
        #               unique_colors = plot_df[color_col].nunique()
        #               if unique_colors <= 20: final_color_col = color_col
        #               else: st.warning(f"Too many categories ({unique_colors}) in '{color_col}' for 3D Scatter color.")

        #          fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=final_color_col,
        #                              title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}" + (f" colored by {final_color_col}" if final_color_col else ""),
        #                              opacity=0.7, template=plotly_template)
        #          fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
        #     else: st.warning(f"No data left for 3D Scatter Plot after handling missing values.")


        elif viz_type == "Parallel Coordinates" and len(numeric_cols) >= 3:
             color_col = all_categoricals[0] if all_categoricals else None
             cols_to_plot = numeric_cols[:8] # Limit dimensions
             cols_to_check = cols_to_plot + ([color_col] if color_col else [])
             plot_df = handle_missing_values(df, cols_to_check, strategy='drop')

             if not plot_df.empty:
                 line_args = {}
                 final_color_col = None

                 if color_col and plot_df[color_col].nunique() <= 15:
                     final_color_col = color_col
                     plot_df[final_color_col] = plot_df[final_color_col].astype('category')
                     color_codes = plot_df[final_color_col].cat.codes
                     color_scale = px.colors.qualitative.Plotly
                     tickvals = list(range(plot_df[final_color_col].nunique()))
                     ticktext = list(plot_df[final_color_col].cat.categories)
                     line_args = dict(
                         color=color_codes,
                         colorscale=color_scale,
                         showscale=True,
                         colorbar=dict(title=final_color_col, tickvals=tickvals, ticktext=ticktext)
                     )
                 else: # Default line color
                     line_args = dict(color='rgba(0,100,200,0.5)') # Semi-transparent blue
                     if color_col: st.warning(f"Too many categories in '{color_col}' for Parallel Coordinates color.")

                 dimensions = [go.parcoords.Dimension(label=col, values=plot_df[col]) for col in cols_to_plot]
                 fig = go.Figure(data=go.Parcoords(line=line_args, dimensions=dimensions))
                 # Apply template after creation for go.Figure
                 fig.update_layout(title=f"Parallel Coordinates Plot ({', '.join(cols_to_plot)})" + (f" colored by {final_color_col}" if final_color_col else ""),
                                   template=plotly_template)
             else: st.warning(f"No data left for Parallel Coordinates Plot after handling missing values.")


        elif viz_type == "Scatter Plot (Colored by Category)" and len(numeric_cols) >= 2 and all_categoricals:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            color_col = all_categoricals[0]
            cols_to_check = [x_col, y_col, color_col]
            plot_df = handle_missing_values(df, cols_to_check, strategy='drop')
            if not plot_df.empty:
                unique_cats = plot_df[color_col].nunique()
                if unique_cats > 20:
                     top_cats = plot_df[color_col].value_counts().nlargest(20).index
                     plot_df = plot_df[plot_df[color_col].isin(top_cats)]
                     st.warning(f"Showing Scatter Plot colored by top 20 categories of '{color_col}'.")
                fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col,
                                 title=f"Scatter Plot: {y_col} vs {x_col}, colored by {color_col}",
                                 opacity=0.7, template=plotly_template)
                # Add trendline per category if meaningful (optional, can be slow)
                # fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            else: st.warning(f"No data left for Colored Scatter Plot after handling missing values.")


        elif viz_type == "Grouped Box Plot" and numeric_cols and len(all_categoricals) >= 2:
            num_col = numeric_cols[0]
            cat_col1 = all_categoricals[0] # X-axis category
            cat_col2 = all_categoricals[1] # Grouping/coloring category
            cols_to_check = [num_col, cat_col1, cat_col2]
            plot_df = handle_missing_values(df, cols_to_check, strategy='drop')
            if not plot_df.empty:
                unique_cats1 = plot_df[cat_col1].nunique()
                unique_cats2 = plot_df[cat_col2].nunique()
                if unique_cats1 > 20 or unique_cats2 > 10:
                     st.warning(f"Too many categories for Grouped Box Plot ({unique_cats1}x{unique_cats2}).")
                else:
                     fig = px.box(plot_df, x=cat_col1, y=num_col, color=cat_col2,
                                  title=f"Grouped Box Plot: {num_col} by {cat_col1}, grouped by {cat_col2}",
                                  template=plotly_template)
            else: st.warning(f"No data left for Grouped Box Plot after handling missing values.")


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

