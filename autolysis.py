import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
import requests
import chardet

def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        print(f"Detected encoding: {result['encoding']}")
        data = pd.read_csv(file_path,encoding=result['encoding'],encoding_errors='replace')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def basic_analysis(data):
    summary = {
        "shape": data.shape,        
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_statistics": data.describe(include='all').to_dict()
    }
    return summary
def select_best_columns(data, target=None, max_columns=3):
    """
    Selects the most relevant columns based on variance or correlation with the target.

    Parameters:
        data (pd.DataFrame): The dataset to analyze.
        target (str): Optional. The target variable to correlate with.
        max_columns (int): The number of columns to select.

    Returns:
        list: List of selected column names.
    """
    numeric_data = data.select_dtypes(include=[np.number])

    if target and target in numeric_data.columns:
        # Calculate correlation with the target
        correlations = numeric_data.corr()[target].abs().sort_values(ascending=False)
        selected_columns = correlations.index.drop(target).tolist()[:max_columns]
    else:
        # Use variance as the selection criterion
        variances = numeric_data.var().sort_values(ascending=False)
        selected_columns = variances.index.tolist()[:max_columns]

    return selected_columns

def generate_correlation_heatmap(data, output_path):
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()

def generate_distribution_plots(data, folder_path, selected_columns):
    """
    Generates distribution plots for the selected columns.

    Parameters:
        data (pd.DataFrame): The dataset to analyze.
        folder_path (str): Path to save the charts.
        selected_columns (list): List of column names to plot.

    Returns:
        None
    """
    for column in selected_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        file_path = os.path.join(folder_path, f"distribution_{column}.png")
        plt.savefig(file_path)
        plt.close()

def query_llm(prompt):
    api_key = os.getenv("AIPROXY_TOKEN")
    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a data analysis assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        # Parse the JSON response
        response_json = response.json()
        print("Response content:", response.json())

        # Check for errors in the response
        if "error" in response_json:
            print(f"API returned an error: {response_json['error']}")
            return ""

        # Access the content safely
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    except requests.RequestException as e:
        print(f"Network error querying LLM: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error querying LLM: {e}")
        return ""
def gather_context(data, filename):
    """
    Gathers contextual information about the dataset.

    Parameters:
        data (pd.DataFrame): The dataset to analyze.
        filename (str): The filename of the dataset.

    Returns:
        dict: Context for LLM interaction.
    """
    context = {
        "filename": filename,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "summary": data.describe(include="all").to_dict(),
    }
    return context

def interact_with_llm(task_type, data_context):
    """
    Interacts with the LLM to get insights, code, or suggestions based on the task type.

    Parameters:
        task_type (str): The task for the LLM ('code', 'summary', 'function_call').
        data_context (dict): Context about the data, including filename, columns, stats, etc.

    Returns:
        str: Response from the LLM.
    """
    prompt = ""
    if task_type == "code":
        prompt = (
            f"You are a Python data analysis assistant. I have the following dataset:\n"
            f"Filename: {data_context['filename']}\n"
            f"Columns: {data_context['columns']}\n"
            f"Data types: {data_context['dtypes']}\n"
            f"Summary statistics: {data_context['summary']}\n"
            f"Generate Python code to analyze this dataset further and provide insights."
        )
    elif task_type == "summary":
        prompt = (
            f"You are a data analyst. Summarize the insights from this dataset:\n"
            f"Filename: {data_context['filename']}\n"
            f"Columns: {data_context['columns']}\n"
            f"Data types: {data_context['dtypes']}\n"
            f"Summary statistics: {data_context['summary']}\n"
        )
    elif task_type == "function_call":
        prompt = (
            f"You are a data scientist. I have the following dataset:\n"
            f"Filename: {data_context['filename']}\n"
            f"Columns: {data_context['columns']}\n"
            f"Data types: {data_context['dtypes']}\n"
            f"Summary statistics: {data_context['summary']}\n"
            f"Suggest specific Python function calls or analyses that can extract more insights."
        )
    else:
        raise ValueError("Invalid task type specified.")

    try:
        response = query_llm(prompt)  # Use the previously defined LLM interaction function
        return response
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def generate_readme(data, summary, charts, llm_insights,charts_folder, output_path):
    with open(charts_folder+"\\"+output_path, "w") as f:
        f.write("# Automated Data Analysis\n\n")
        f.write("## Dataset Overview\n")
        f.write(f"Shape: {summary['shape']}\n\n")
        f.write("### Columns and Data Types\n")
        for col, dtype in summary["columns"].items():
            f.write(f"- {col}: {dtype}\n")

        f.write("\n### Missing Values\n")
        for col, missing in summary["missing_values"].items():
            f.write(f"- {col}: {missing} missing values\n")

        f.write("\n## Insights from the LLM\n")
        f.write(f"{llm_insights}\n\n")

        f.write("## Visualizations\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")

def main():
    if len(sys.argv) != 2:
       print("Usage: uv run autolysis.py <dataset.csv>")
       sys.exit(1)

    load_dotenv()

    file_path = sys.argv[1]#'goodreads.csv' #sys.argv[1]
    data = load_data(file_path)

    print("Performing basic analysis...")
    summary = basic_analysis(data)
    
    # Create a folder named after the dataset (without extension)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    charts_folder = dataset_name
    os.makedirs(charts_folder, exist_ok=True)
    
    print("Selecting the best columns for visualization...")
    target_column = None  # Replace with the target column name if applicable
    best_columns = select_best_columns(data, target=target_column, max_columns=3)
    
    print("Generating visualizations...")
    heatmap_path = os.path.join(charts_folder, "correlation_heatmap.png")
    generate_correlation_heatmap(data, heatmap_path)
    generate_distribution_plots(data, charts_folder, best_columns)

    print("Consulting the LLM for insights...")
    llm_prompt = (
        f"You are a data analyst. Summarize the insights from this dataset:\n"
            f"Filename: {file_path}\n"
            f"Columns and Data Types: {summary['columns']}\n"           
            f"Sample data:{data.iloc[0]}\\n"
            f"Summary statistics: {summary['summary_statistics']}\n"
    )
    llm_insights = query_llm(llm_prompt)

    print("Generating README.md...")
    charts = [heatmap_path] + [os.path.join(charts_folder, f"distribution_{col}.png") for col in data.select_dtypes(include=[np.number]).columns]
    generate_readme(data, summary, charts, llm_insights,charts_folder, "README.md")

    print(f"Analysis complete. Results saved to README.md and visualizations saved in the '{charts_folder}' folder.")
    
if __name__ == "__main__":
    main()
