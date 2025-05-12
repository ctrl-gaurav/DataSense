import openai 
import pandas as pd
from openai import OpenAI


openai.api_key = ""
client = OpenAI(api_key="")


def extract_dataset_info(df):
    summary_data = []

    for col in df.columns:
        dtype = df[col].dtype
        col_info = {
            "Column": col,
            "Data Type": str(dtype)
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["Mean"] = round(df[col].mean(), 3)
            col_info["Std Dev"] = round(df[col].std(), 3)
            col_info["Max"] = round(df[col].max(), 3)
            col_info["Min"] = round(df[col].min(), 3)
        else:
            unique_vals = df[col].dropna().unique()
            col_info["Unique Count"] = len(unique_vals)
            if len(unique_vals) <= 5:
                col_info["Unique Values"] = list(unique_vals)

        summary_data.append(col_info)

    dataset_desc = ""
    dataset_desc += "\tColumn Names and Data Types - \n"
    for data in summary_data:
        dataset_desc += f"\t\t{data['Column']} ({data['Data Type']})\n"

    dataset_desc += "\tDataset Statistics:\n"
    for data in summary_data:
        if data['Data Type'] == "category" or data['Data Type'] == str or data['Data Type'] == "object":
            if data["Unique Count"] < 5:
                dataset_desc += f"\t\t{data['Column']}: total {data['Unique Count']} unique values, those are {', '.join(data['Unique Values'])}\n"
            else:
                dataset_desc += f"\t\t{data['Column']}: total {data['Unique Count']} unique values\n"
        else:
                dataset_desc += f"\t\t{data['Column']}: mean -> {data['Mean']}, standard deviation -> {data['Std Dev']}, Maximum -> {data['Max']}, Minimum -> {data['Min']}\n"
    dataset_desc += "Sample Rows:\n"
    rows = df.iloc[:5].to_csv(index=False)
    dataset_desc += rows

    return dataset_desc

def extract_plot_info(PLOT):
    
    desc = '''
    
    '''
    return desc

def info_loader(df_path, PLOT):
    prompt = open("./prompt.md", "r").read()
    df = pd.read_csv(df_path)
    df_desc = extract_dataset_info(df)
    plot_desc = extract_plot_info(PLOT)
    return prompt, df_desc, plot_desc

def load_dummy_data():
    prompt = open("./prompt.md", "r").read()
    df_desc = open("./dummy_df_desc.md").read()
    plot_desc = open("./dummy_plot_desc.md").read()
    return prompt, df_desc, plot_desc


# prompt, df_desc, plot_desc = info_loader("path_to_csv", "YOUR_PLOT_INFO")

prompt, df_desc, plot_desc = load_dummy_data()


messages = [
    {
        "role": "system",
        "content": prompt,
    },
    {
        "role": "user",
        "content": f"Dataset Description: \n\t{df_desc}\n\n\Plot Description: \n\t{plot_desc}",
    }
]

response = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="medium",
        messages=messages)

print(response.choices[0].message.content)
