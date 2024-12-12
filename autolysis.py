# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "seaborn",
#   "matplotlib",
#   "openai",
#   "python-dotenv",
#   "requests",
#   "chardet",
#   "scikit-learn",
#   "statsmodels",
#   "scipy"
# ]
# ///
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
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.metrics import silhouette_score

def load_data(file_path):
    """
    Loads a CSV file into a Pandas DataFrame while detecting and handling its character encoding.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        DataFrame: The loaded dataset.

    Exceptions:
        Exits the program with an error message if the file cannot be loaded.
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        print(f"Detected encoding: {result['encoding']}")
        data = pd.read_csv(file_path,encoding=result['encoding'],encoding_errors='replace')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

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
    """
    Generates a heatmap showing the correlation between numeric columns in the dataset.

    Args:
        data (DataFrame): The dataset to analyze. Only numeric columns are considered.
        output_path (str): The file path to save the generated heatmap image.

    Returns:
        DataFrame: The correlation matrix used to generate the heatmap.
    """
    # Select only numeric columns from the dataset for correlation analysis
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Compute the correlation matrix for the selected numeric data
    correlation_matrix = numeric_data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f",
        xticklabels=numeric_data.columns, 
        yticklabels=numeric_data.columns
    )
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    
    # Save the heatmap as an image
    plt.savefig(output_path)
    plt.close()
    
    return correlation_matrix

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
        "shape": data.shape,  
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary": data.describe(include="all").to_dict(),
    }
    return context
def query_llm(prompt):
    """
    Sends a prompt to the LLM API and retrieves the response.

    Args:
        prompt (str): The prompt to send to the LLM for analysis.

    Returns:
        str: The content of the LLM's response, or an empty string if an error occurs.
    """
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
def interact_with_llm(task_type, data_context):
    """
    Interacts with the LLM to get insights, code, or suggestions based on the task type.
    Based on the task type, the prompt is sent to LLM for summary/code/function call details
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
            #f"Data types: {data_context['dtypes']}\n"
            f"Summary statistics: {data_context['summary']}\n"
            f"Generate Python code to analyze this dataset further. Do not add anything other than python code. \\n"
            f"Include code to detect encoding of the csv file and use the same encoding"
            f"Dataset may have non-numeric. Exclude them while performing numeric analysis"
        )
    elif task_type == "summary":
        prompt = (
            f"You are a data analyst. Summarize the insights from this dataset:\n"
            f"Filename: {data_context['filename']}\n"
            f"Columns: {data_context['columns']}\n"
            #f"Data types: {data_context['dtypes']}\n"
            f"Summary statistics: {data_context['summary']}\n"
        )
    elif task_type == "function_call":
        prompt = (
            f"You are a data scientist. I have the following dataset:\n"
            f"Filename: {data_context['filename']}\n"
            f"Columns: {data_context['columns']}\n"
            #f"Data types: {data_context['dtypes']}\n"
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

def generate_distribution_plots(data, output_path, selected_columns):    
    """
    Generates a distribution chart for a given numeric column in the dataset.

    Args:
        data (DataFrame): The dataset containing the column.
        column (str): The column to analyze.
        output_path (str): The file path to save the generated chart image.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data[selected_columns], kde=True, bins=30, color='blue', label=f"Distribution of {selected_columns}")
    plt.title(f"Distribution of {selected_columns}")
    plt.xlabel(selected_columns)
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')  # Adding a legend for clarity
    plt.savefig(output_path)
    plt.close()

# "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",https://api.openai.com/v1/chat/completions
def generate_readme(data, summary, charts, llm_insights,output_path):
    """
    Generates a README.md file summarizing the results of data analysis.

    Args:
        data (DataFrame): The dataset being analyzed.
        summary (dict): Summary statistics and metadata about the dataset.
        charts (list): List of file paths to generated visualizations.
        llm_insights (str): Analytical insights provided by the LLM.
        output_path (str): The path where the README file will be saved.

    Returns:
        None
    """
    with open(output_path, "w") as f:
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
        f.write("1. 'correlation_heatmap.png': Correlation heatmap of features.\n")
        f.write("2. 'pca_explained_variance.png': PCA variance explanation.\n")
        f.write("3. 'cluster_visualization.png': Cluster analysis scatterplot.\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")

def main():
    if len(sys.argv) != 2:
       print("Usage: uv run autolysis.py <dataset.csv>")
       sys.exit(1)

    load_dotenv()

    file_path = sys.argv[1]#'goodreads.csv' #sys.argv[1]
    data = load_data(file_path)
    # Create a folder named after the dataset (without extension)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    charts_folder = dataset_name
    os.makedirs(charts_folder, exist_ok=True)
    
    #data_cleaned = data.dropna()
    # Separate numeric and categorical columns
    numeric_data = data.select_dtypes(include=[np.number])
    categorical_data = data.select_dtypes(exclude=[np.number])

    # Impute numeric data with the mean
    numeric_imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = pd.DataFrame(numeric_imputer.fit_transform(numeric_data), 
                                        columns=numeric_data.columns)

    # Impute categorical data with the most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_data_imputed = pd.DataFrame(categorical_imputer.fit_transform(categorical_data), 
                                            columns=categorical_data.columns)

    # Combine the data back together
    data_imputed = pd.concat([numeric_data_imputed, categorical_data_imputed], axis=1)
    
    print("Gathering dataset context...")
    summary = gather_context(data_imputed, os.path.basename(file_path))
    
    print("Selecting the best columns for visualization...")
    target_column = None  # Replace with the target column name if applicable
    best_columns = select_best_columns(data_imputed, target=target_column, max_columns=3)
    
    print("Generating visualizations...")
    heatmap_path = os.path.join(charts_folder, "correlation_heatmap.png")
    correlation_matrix = generate_correlation_heatmap(data_imputed, heatmap_path)
    correlations = correlation_matrix.unstack().reset_index()
    correlations.columns = ['Variable1', 'Variable2', 'Correlation']
    correlations = correlations[correlations['Variable1'] != correlations['Variable2']]  # Remove self-correlations
    correlations['AbsCorrelation'] = correlations['Correlation'].abs()
    top_correlations = correlations.sort_values(by='AbsCorrelation', ascending=False).head(3)

    generate_distribution_plots(data_imputed, charts_folder, best_columns)

     # Ask the LLM to summarize the dataset
    print("Asking LLM for a summary of the dataset...")
    summary_response = interact_with_llm("summary", summary)
    #print("Summary from LLM:")
    #print(summary_response)

    # Ask the LLM to suggest Python code for further analysis
    print("Asking LLM for Python code suggestions...")
    code_response = interact_with_llm("code", summary)
    print("Code from LLM:")
    print(code_response)

    # Execute the suggested code (with caution)
    try:
        print("Executing LLM-generated code...")
        exec(code_response)
    except Exception as e:
        print(f"Error executing LLM-generated code: {e}")

    # Ask the LLM for specific function calls
    print("Asking LLM for additional function call suggestions...")
    functions_response = interact_with_llm("function_call", summary)
    print("Function call suggestions from LLM:")
    print(functions_response)

    print("Generating README.md...")
    charts = [heatmap_path] + [os.path.join(charts_folder, f"distribution_{col}.png") for col in data.select_dtypes(include=[np.number]).columns]
    readme_path = os.path.join(charts_folder, "README.md")
    generate_readme(data_imputed, summary, charts, summary_response, readme_path)

    print(f"Analysis complete. Results saved to README.md and visualizations saved in the '{charts_folder}' folder.")
    
if __name__ == "__main__":
    main()
