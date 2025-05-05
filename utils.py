import pandas as pd
import numpy as np
import io
import json
from sklearn import datasets
import streamlit as st
import yaml
from typing import Dict, Any, List, Optional
import re # Import the 're' module

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file '{config_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

def get_sklearn_datasets() -> Dict[str, callable]:
    return {
        'Iris': datasets.load_iris,
        'Wine': datasets.load_wine,
        'Breast Cancer': datasets.load_breast_cancer,
        'Diabetes': datasets.load_diabetes,
        'California Housing': datasets.fetch_california_housing,
        'Digits': datasets.load_digits
    }

def load_sklearn_dataset(name: str) -> Optional[pd.DataFrame]:
    sklearn_datasets = get_sklearn_datasets()
    if name in sklearn_datasets:
        try:
            # The Breast Cancer dataset has feature names with spaces and parentheses
            # which might have triggered the regex issue more easily.
            dataset = sklearn_datasets[name]()

            # Handle potential differences in dataset structure (e.g., fetch_california_housing)
            if isinstance(dataset, dict) and 'data' in dataset and 'feature_names' in dataset:
                 data = dataset['data']
                 feature_names = dataset['feature_names']
                 target = dataset.get('target')
                 target_names = dataset.get('target_names')
            elif hasattr(dataset, 'data') and hasattr(dataset, 'feature_names'):
                 data = dataset.data
                 feature_names = dataset.feature_names
                 target = getattr(dataset, 'target', None)
                 target_names = getattr(dataset, 'target_names', None)
            else:
                 st.error(f"Could not determine data structure for dataset '{name}'.")
                 return None

            df = pd.DataFrame(data, columns=feature_names)

            if target is not None:
                target_name = 'target'
                # Avoid potential duplicate column names
                original_target_name = target_name
                count = 1
                while target_name in df.columns:
                    target_name = f"{original_target_name}_{count}"
                    count += 1

                if target_names is not None and len(target_names) > 0 and target.ndim == 1:
                     # Check if target is suitable for categorical conversion
                     if np.issubdtype(target.dtype, np.integer) and np.all(target >= 0) and np.max(target) < len(target_names):
                         df[target_name] = pd.Categorical.from_codes(target, target_names)
                     else:
                         df[target_name] = target # Keep as is if not standard classification target
                else:
                    df[target_name] = target
            return df
        except Exception as e:
            st.error(f"Error loading scikit-learn dataset '{name}': {e}")
            import traceback
            st.code(traceback.format_exc()) # Show traceback for debugging
            return None
    else:
        st.error(f"Unknown scikit-learn dataset: {name}")
        return None

def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is not None:
        try:
            # Read content based on file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                # Try reading line-delimited JSON first, then standard JSON array
                try:
                    # Use BytesIO for consistent reading
                    bytes_data = uploaded_file.getvalue()
                    df = pd.read_json(io.BytesIO(bytes_data), lines=True)
                except ValueError:
                    # Reset file pointer if using original object (getvalue avoids this need)
                    # uploaded_file.seek(0)
                    df = pd.read_json(io.BytesIO(bytes_data)) # Read as standard JSON
            elif uploaded_file.name.endswith('.txt'):
                # Assuming tab-separated for .txt, adjust if needed
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                st.error("Unsupported file type. Please upload CSV, JSON, or TXT.")
                return None

            # Basic cleaning: remove fully empty rows/columns
            df.dropna(axis=0, how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            if df.empty:
                st.warning("The uploaded file is empty or contains only empty values after cleaning.")
                return None

            return df
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
            return None
        except Exception as e:
            st.error(f"Error reading file '{uploaded_file.name}': {e}")
            import traceback
            st.code(traceback.format_exc()) # Show traceback for debugging
            return None
    return None

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans DataFrame column names for better compatibility."""
    new_columns = {}
    existing_new_names = set() # Keep track of generated names to ensure uniqueness

    for col in df.columns:
        original_col_str = str(col) # Ensure it's a string
        new_col = original_col_str.strip()

        # Replace sequences of non-alphanumeric characters (excluding underscore) with a single underscore
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', new_col)

        # Remove leading/trailing underscores that might result from replacement
        new_col = new_col.strip('_')

        # Ensure column name is not empty after cleaning
        if not new_col:
             new_col = f"column_{np.random.randint(1000)}" # Assign a random name

        # Ensure uniqueness among the *new* column names being generated
        count = 1
        original_new_col = new_col
        while new_col in existing_new_names:
            new_col = f"{original_new_col}_{count}"
            count += 1

        new_columns[col] = new_col
        existing_new_names.add(new_col) # Add the finalized new name to the set

    df.rename(columns=new_columns, inplace=True)
    return df


def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df is None:
         return None
    df_copy = df.copy() # Work on a copy to avoid modifying original session state df
    df_copy = clean_column_names(df_copy)

    # Attempt to convert object columns to numeric or datetime
    for col in df_copy.select_dtypes(include=['object']).columns:
        # Try numeric first
        try:
            converted_col = pd.to_numeric(df_copy[col])
            # Check if conversion resulted in all NaNs, if so, maybe it wasn't numeric
            if not converted_col.isnull().all():
                 df_copy[col] = converted_col
                 continue # Skip to next column if numeric conversion successful
        except (ValueError, TypeError):
            pass # Ignore errors if not numeric

        # Try datetime conversion if not numeric
        try:
            # Try datetime conversion with inference, handle potential format issues
            df_copy[col] = pd.to_datetime(df_copy[col], errors='raise', infer_datetime_format=True)
            continue # Skip to next column if datetime conversion successful
        except (ValueError, TypeError, OverflowError): # Added OverflowError
            pass # Ignore errors if not datetime

        # If still object, check if it looks like boolean (case-insensitive)
        # Increase nunique check slightly, handle NaN explicitly
        unique_vals_no_nan = df_copy[col].dropna().unique()
        if 0 < len(unique_vals_no_nan) <= 5: # Check for low cardinality, potentially boolean-like
             bool_like = all(str(v).lower() in ['true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'] for v in unique_vals_no_nan)
             if bool_like:
                 try:
                    # Attempt boolean conversion (handle various string representations)
                    bool_map = {
                        'true': True, 'false': False, 'yes': True, 'no': False,
                        '1': True, '0': False, 't': True, 'f': False, 'y': True, 'n': False,
                         # Handle actual booleans/numerics if read as object initially
                         1: True, 0: False, True: True, False: False
                    }
                    # Apply mapping robustly, keeping NaNs
                    # Convert to string and lowercase for mapping lookup
                    df_copy[col] = df_copy[col].apply(lambda x: bool_map.get(str(x).lower(), pd.NA) if pd.notna(x) else pd.NA)
                    df_copy[col] = df_copy[col].astype('boolean') # Use pandas nullable boolean type
                 except Exception:
                    pass # Ignore if boolean conversion fails

    # Convert Int64 (nullable int) columns with no NaNs to standard int
    for col in df_copy.select_dtypes(include=['Int64']).columns:
         if not df_copy[col].isnull().any():
             df_copy[col] = df_copy[col].astype(int)

    # Convert Float64 columns with no NaNs and all integer values to standard int
    for col in df_copy.select_dtypes(include=['float64']).columns:
         # Check column exists and is not empty after potential drops
         if col in df_copy.columns and not df_copy[col].isnull().all():
             # Drop NaNs for the integer check, but perform on original float column
             col_no_nan = df_copy[col].dropna()
             if not col_no_nan.empty:
                 try:
                     # Check if all non-NaN values are integers
                     is_integer = (col_no_nan == col_no_nan.astype(int)).all()
                     # Only convert if *all* original values (including potential NaNs) allow it
                     if is_integer and not df_copy[col].isnull().any():
                         df_copy[col] = df_copy[col].astype(int)
                 except Exception:
                     pass # Ignore potential errors during conversion check

    return df_copy

def analyze_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or df.empty:
         st.warning("Cannot analyze an empty DataFrame.")
         # Return a default structure to avoid errors downstream
         return {
            "num_rows": 0, "num_columns": 0, "columns": [],
            "column_types": {}, "column_stats": {}, "missing_values": {},
            "correlations": None, "sample_data": None,
            "potential_categorical": [], "potential_numeric": [],
            "potential_datetime": [], "potential_text": [], "potential_boolean": [],
         }

    data_cfg = config.get('data_settings', {})
    max_rows_preview = data_cfg.get('max_rows_preview', 5)
    max_str_len = data_cfg.get('max_string_length_preview', 50)
    corr_threshold = data_cfg.get('numeric_correlation_threshold', 0.5)
    cat_threshold_ratio = data_cfg.get('categorical_threshold_ratio', 0.5) # Ratio unique/rows
    max_unique_for_categorical = 50 # Absolute max unique values for auto-categorical
    max_cat_values_display = data_cfg.get('max_categorical_values_display', 10)

    metadata = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "column_types": {},
        "column_stats": {},
        "missing_values": {},
        "correlations": None,
        "sample_data": None,
        "potential_categorical": [],
        "potential_numeric": [],
        "potential_datetime": [],
        "potential_text": [],
        "potential_boolean": [],
    }

    # Sample data handling potential errors during JSON conversion
    try:
        sample_df = df.head(max_rows_preview).copy()
        # Truncate long strings in sample data robustly
        for col in sample_df.columns:
             if pd.api.types.is_string_dtype(sample_df[col]) or sample_df[col].dtype == 'object':
                 sample_df[col] = sample_df[col].astype(str).str[:max_str_len]
        # Convert to dict, handling potential non-serializable types gracefully
        metadata["sample_data"] = json.loads(sample_df.to_json(orient="records", default_handler=str))
    except Exception as e:
         st.warning(f"Could not serialize sample data preview: {e}")
         # Fallback if conversion fails
         metadata["sample_data"] = df.head(max_rows_preview).to_string()


    for col in df.columns:
        col_data = df[col]
        col_type_str = str(col_data.dtype)
        # Drop NaNs for unique count unless column is entirely NaN
        unique_count = col_data.nunique() if not col_data.isnull().all() else 0
        missing_count = int(col_data.isnull().sum())
        if missing_count > 0:
            metadata["missing_values"][col] = missing_count

        col_stats = {"unique_values": int(unique_count)}
        non_nan_count = len(df) - missing_count

        # Type Inference Logic
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            metadata["column_types"][col] = "numeric"
            metadata["potential_numeric"].append(col)
            if non_nan_count > 0:
                 desc = col_data.describe()
                 col_stats.update({
                    "min": float(desc.get('min', np.nan)),
                    "max": float(desc.get('max', np.nan)),
                    "mean": float(desc.get('mean', np.nan)),
                    "median": float(desc.get('50%', np.nan)), # Use '50%' for median from describe()
                    "std": float(desc.get('std', np.nan)),
                 })
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            metadata["column_types"][col] = "datetime"
            metadata["potential_datetime"].append(col)
            if non_nan_count > 0:
                 col_stats.update({
                    "min": str(col_data.min()),
                    "max": str(col_data.max()),
                 })
        elif pd.api.types.is_bool_dtype(col_data) or col_type_str == 'boolean':
             metadata["column_types"][col] = "boolean"
             metadata["potential_boolean"].append(col)
             if non_nan_count > 0:
                 value_counts = col_data.value_counts().to_dict()
                 col_stats["value_counts"] = {str(k): int(v) for k, v in value_counts.items()}
        # Categorical Check: Explicit categorical type OR (low unique count relative to rows AND low absolute unique count)
        elif pd.api.types.is_categorical_dtype(col_data) or \
             (non_nan_count > 0 and unique_count / non_nan_count < cat_threshold_ratio and unique_count <= max_unique_for_categorical) or \
             (non_nan_count > 0 and unique_count <= 5): # Also consider very low unique counts as categorical regardless of ratio
            metadata["column_types"][col] = "categorical"
            metadata["potential_categorical"].append(col)
            if non_nan_count > 0:
                 value_counts = col_data.value_counts()
                 top_values = value_counts.head(max_cat_values_display).to_dict()
                 col_stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                 if unique_count > max_cat_values_display:
                     col_stats["top_values"]["...others"] = int(value_counts.iloc[max_cat_values_display:].sum())
        else: # Default to text if none of the above
            metadata["column_types"][col] = "text"
            metadata["potential_text"].append(col)
            # Add sample text values if useful
            if non_nan_count > 0:
                 # Get unique non-null values, convert to string, truncate
                 sample_texts = col_data.dropna().unique()[:max_cat_values_display]
                 col_stats["sample_values"] = [str(s)[:max_str_len] for s in sample_texts]

        metadata["column_stats"][col] = col_stats

    # Calculate correlations for numeric columns
    numeric_cols = metadata["potential_numeric"]
    if len(numeric_cols) > 1:
        try:
            # Select only numeric columns that actually exist in the df
            valid_numeric_cols = [col for col in numeric_cols if col in df.columns]
            if len(valid_numeric_cols) > 1:
                 # Compute correlation, handle potential errors if columns have no variance
                 corr_matrix = df[valid_numeric_cols].corr(numeric_only=True).round(2)
                 correlations = {}
                 for i, col1 in enumerate(valid_numeric_cols):
                     for j, col2 in enumerate(valid_numeric_cols):
                         if i < j: # Avoid duplicates and self-correlation
                             if col1 in corr_matrix.index and col2 in corr_matrix.columns:
                                 corr_val = corr_matrix.loc[col1, col2]
                                 # Check corr_val is not NaN before comparing
                                 if pd.notna(corr_val) and abs(corr_val) >= corr_threshold:
                                     if col1 not in correlations:
                                         correlations[col1] = {}
                                     correlations[col1][col2] = float(corr_val)
                 if correlations: # Only add if there are significant correlations
                      metadata["correlations"] = correlations
        except Exception as e:
            print(f"Could not compute correlations: {e}") # Log error but continue

    return metadata

def generate_visualization_options(metadata: Dict[str, Any]) -> List[str]:
    options = set()
    num_cols = metadata.get("num_columns", 0)
    if num_cols == 0: return ["Table"] # Handle empty metadata case

    numeric = metadata.get("potential_numeric", [])
    categorical = metadata.get("potential_categorical", [])
    datetime = metadata.get("potential_datetime", [])
    boolean = metadata.get("potential_boolean", [])
    text = metadata.get("potential_text", [])

    all_categoricals = categorical + boolean # Treat booleans like categoricals for plotting

    # Single column visualizations
    if numeric:
        options.add("Histogram")
        options.add("Box Plot")
        options.add("Density Plot") # Requires scipy
    if all_categoricals:
        # Only suggest pie chart for low cardinality
        cat_col_for_pie = all_categoricals[0] if all_categoricals else None
        if cat_col_for_pie and metadata.get("column_stats", {}).get(cat_col_for_pie, {}).get("unique_values", 100) <= 15:
             options.add("Pie Chart")
        options.add("Bar Chart (Counts)")
    if datetime:
         # Could add time series decomposition if relevant, but keep simple for now
         pass

    # Two column visualizations
    if len(numeric) >= 2:
        options.add("Scatter Plot")
        # Only add heatmap if correlations were likely computed and significant
        if metadata.get("correlations") is not None and metadata["correlations"]: # Check if correlations dict is not empty
             options.add("Heatmap (Correlation)")
    if numeric and all_categoricals:
        options.add("Box Plot by Category")
        options.add("Violin Plot by Category")
        options.add("Bar Chart (Aggregated)") # e.g., mean/sum of numeric per category
    if numeric and datetime:
        options.add("Line Chart")
        options.add("Area Chart")
    if all_categoricals and datetime:
         # Requires specific aggregation logic, maybe add later if needed
         # options.add("Stacked Bar Chart (Counts over Time)")
         pass
    if len(all_categoricals) >= 2:
         # Heatmap for counts might be too sparse often, consider carefully
         # options.add("Heatmap (Counts)")
         options.add("Stacked Bar Chart")
         options.add("Grouped Bar Chart")

    # Three+ column visualizations
    if len(numeric) >= 3:
        options.add("3D Scatter Plot")
        options.add("Parallel Coordinates")
    if len(numeric) >= 2 and all_categoricals:
        options.add("Scatter Plot (Colored by Category)")
    if numeric and len(all_categoricals) >= 2:
         options.add("Grouped Box Plot")

    # Always include table as an option
    options.add("Table")

    # Handle cases with only text data (maybe add later)
    # if not numeric and not all_categoricals and not datetime and text:
    #     options.add("Word Cloud")

    # Ensure some options exist
    if not options:
        return ["Table"]

    # Sort for consistency
    return sorted(list(options))


def create_llm_prompt(df_name: str, metadata: Dict[str, Any], viz_options: List[str], config: Dict[str, Any]) -> str:
    """Creates the prompt for the LLM, requesting detailed insights and story."""
    data_cfg = config.get('data_settings', {})
    max_str_len = data_cfg.get('max_string_length_preview', 50)

    prompt = f"You are an expert data analyst and storyteller. Analyze the dataset '{df_name}' and recommend the single BEST visualization type to reveal key insights.\n\n"
    prompt += "**Dataset Summary:**\n"
    prompt += f"- Rows: {metadata.get('num_rows', 'N/A')}, Columns: {metadata.get('num_columns', 'N/A')}\n"
    columns_list = metadata.get('columns', [])
    if columns_list:
        prompt += f"- Column Names: {', '.join(columns_list)}\n\n"
    else:
        prompt += "- Column Names: Not available\n\n"


    prompt += "**Column Details:**\n"
    if not columns_list:
         prompt += "No column details available.\n\n"
    else:
        for col in columns_list:
            col_type = metadata.get('column_types', {}).get(col, 'unknown')
            stats = metadata.get('column_stats', {}).get(col, {})
            missing = metadata.get('missing_values', {}).get(col, 0)
            missing_str = f" ({missing} missing)" if missing > 0 else ""
            prompt += f"- **{col}** (Type: {col_type}){missing_str}:\n"
            prompt += f"  - Unique Values: {stats.get('unique_values', 'N/A')}\n"

            if col_type == 'numeric':
                # Format numeric stats carefully, handle None/NaN
                stat_parts = []
                for k, v in stats.items():
                    if k != 'unique_values' and v is not None and pd.notna(v):
                        stat_parts.append(f"{k}={v:.2f}")
                if stat_parts:
                     prompt += f"  - Stats: {', '.join(stat_parts)}\n"
            elif col_type == 'categorical' or col_type == 'boolean':
                top_vals = stats.get('top_values', stats.get('value_counts', {}))
                if top_vals:
                     # Ensure keys are strings for display
                     top_vals_str = ", ".join([f"'{str(k)}': {v}" for k, v in top_vals.items()])
                     prompt += f"  - Top Values: {top_vals_str}\n"
            elif col_type == 'datetime':
                dt_min = stats.get('min', 'N/A')
                dt_max = stats.get('max', 'N/A')
                if dt_min != 'N/A' or dt_max != 'N/A':
                    prompt += f"  - Range: {dt_min} to {dt_max}\n"
            elif col_type == 'text':
                 samples = stats.get('sample_values', [])
                 if samples:
                     samples_str = ", ".join([f"'{s}'" for s in samples])
                     prompt += f"  - Sample Values: {samples_str}\n"
            prompt += "\n"


    if metadata.get("correlations"):
        prompt += f"**Significant Numeric Correlations (Threshold > {config.get('data_settings', {}).get('numeric_correlation_threshold', 0.5)}):**\n"
        for col1, corrs in metadata["correlations"].items():
            for col2, val in corrs.items():
                prompt += f"- {col1} & {col2}: {val:.2f}\n"
        prompt += "\n"

    sample_data = metadata.get("sample_data")
    if sample_data:
        prompt += f"**Sample Data (first {config.get('data_settings', {}).get('max_rows_preview', 5)} rows):**\n"
        try:
            # Use pandas to format the sample data nicely if it's a list of dicts
            if isinstance(sample_data, list) and sample_data:
                 sample_df_str = pd.DataFrame(sample_data).to_string(index=False, max_colwidth=max_str_len)
                 prompt += f"```\n{sample_df_str}\n```\n\n"
            else:
                 # Fallback for other formats or errors
                 prompt += f"{json.dumps(sample_data, indent=2, default=str)}\n\n"
        except Exception:
             # Fallback if pandas formatting fails
             prompt += f"{json.dumps(sample_data, indent=2, default=str)}\n\n"


    prompt += "**Available Visualization Options:**\n"
    if viz_options:
        prompt += "- " + "\n- ".join(viz_options) + "\n\n"
    else:
        prompt += "No specific visualization options generated (defaulting to Table).\n\n"


    prompt += "**Your Task:**\n"
    prompt += "1.  **Reasoning:** Explain step-by-step why you are recommending a specific visualization from the list above. What goal does this visualization achieve (e.g., comparing distributions, showing relationships, tracking trends)? Why is it better than other suitable options for *this specific dataset*?\n"
    prompt += "2.  **Structure Response:** Present your analysis in two distinct sections using Markdown headings exactly as follows:\n"
    prompt += "    - **`### Data Insights`**: Provide a detailed summary of key findings from the data. Discuss:\n"
    prompt += "        - Important distributions (e.g., skewed, bimodal) for key numeric variables.\n"
    prompt += "        - Significant correlations (mention specific pairs and strength if available in metadata).\n"
    prompt += "        - Notable patterns in categorical data (e.g., dominant categories, imbalances).\n"
    prompt += "        - Potential outliers or unusual values if apparent from stats.\n"
    prompt += "        - Impact of any missing data mentioned in the metadata.\n"
    prompt += "        - Relationships between variables suggested by the data types and stats.\n"
    prompt += "    - **`### Data Story`**: Weave the key insights into an engaging narrative. Explain *how* the recommended visualization helps to see and understand these specific insights. Tell a short story about what the data reveals, using the visualization as the central element of the explanation. Make it clear and understandable for someone unfamiliar with the dataset.\n"
    prompt += "3.  **Recommendation:** Conclude your *entire* response with the single best visualization type from the provided list.\n\n"

    prompt += "**Output Format Requirements:**\n"
    prompt += "- Use the exact Markdown headings `### Data Insights` and `### Data Story`.\n"
    prompt += "- Provide detailed analysis in the Insights section.\n"
    prompt += "- Craft a clear, narrative story in the Story section linked to the visualization.\n"
    prompt += "- **ABSOLUTELY CRITICAL:** The response MUST end *immediately* after the final closing brace `}` of the boxed answer. There should be NO text, explanation, whitespace, or newlines following it.\n"
    prompt += "    - **Correct End Format:** `...end of story text.\n\\boxed{answer: [Visualization Type]}`\n"
    prompt += "    - **Incorrect End Format:** `...end of story text.\n\\boxed{answer: [Visualization Type]}\nOkay, here is the recommendation.`\n"
    prompt += "    - **Incorrect End Format:** `...end of story text.\nRecommendation:\n\\boxed{answer: [Visualization Type]}`\n"
    prompt += "The final line MUST be the boxed answer itself.\n\n"
    prompt += "**Final Answer Format:**\n"
    prompt += "\\boxed{answer: [Visualization Type]}"


    return prompt
