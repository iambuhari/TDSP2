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
#   "scipy",
#   "tiktoken",
#   "tenacity"
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
import json
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
import tiktoken
import ast
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential

retry_count = 0
max_retries = 3

#https://aiproxy.sanand.workers.dev/openai/v1/chat/completions
#https://api.openai.com/v1/chat/completions
#OPENAIAPIURL='https://api.openai.com/v1/chat/completions'
OPENAIAPIURL='https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

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
            OPENAIAPIURL,#"https://api.openai.com/v1/chat/completions",
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
def interact_with_llm(task_type, datadata_imputed,filename):
    """
    Interacts with the LLM to get insights, code, or suggestions based on the task type.
    Based on the task type, the prompt is sent to LLM for summary/code/function call details
    Parameters:
        task_type (str): The task for the LLM ('code', 'summary', 'function_call').
        data_context (dict): Context about the data, including filename, columns, stats, etc.

    Returns:
        str: Response from the LLM.
    """
    prompt = generate_dynamic_prompt(datadata_imputed,filename)
    if task_type == "code":
        prompt += (
            f"Generate Python code for # 1. Correlation Matrix for numeric features, return correlation_matrix\\n"
            f"# 2. PCA for dimensionality reduction to analyze this dataset further, return pca_result\\n"
            #f"# 3. K-Means Clustering for categorical features, return df['cluster'] \\n"
            f"Generate only python code only. Add code to detect the encoding using chardet.detect and use for opening the file.\\n"  
            f"Filter the non-numeric column from the dataset before performing correlation, pca and k-means analysis\\n"                   
            f"Fill missing values with the most frequence value"
            f"Exclude NaNs data for PCA "
            f"Automatically identifies categorical columns.\\n"
            #f"Encodes them appropriately for K-Means clustering (e.g., using One-Hot or Label Encoding)\\n"
            #f"Performs K-Means clustering on the processed DataFrame.\\n"
            #f"Handles missing values gracefully.\\n"
        )
    elif task_type == "summary":
        prompt += (
           f"Provide insights and suggest three analyses in no more than 100 words."
        )
    elif task_type == "function_call":
        prompt += (
            f"Suggest top 3 specific Python function calls or analyses that can extract more insights"
        )
    else:
        raise ValueError("Invalid task type specified.")

    try:
        response = query_llm(prompt)  # Use the previously defined LLM interaction function
        return response
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None
def choose_analysis_methods(data):
    """
    Dynamically selects analysis methods based on data characteristics.

    Args:
        data (DataFrame): The dataset to analyze.

    Returns:
        list: A list of method names to apply.
    """
    methods = []
    if not data.isnull().values.any():
        methods.append("Correlation Analysis")
    if len(data.select_dtypes(include=[np.number]).columns) > 3:
        methods.append("PCA for dimensionality reduction")
    if data.shape[0] > 500:
        methods.append("Outlier detection using Isolation Forest")
    if len(data.select_dtypes(exclude=[np.number]).columns) > 2:
        methods.append("Clustering on categorical data")

    return methods

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def execute_code_with_tenacity(code):
    
    try:
        result_prompt = """ 
# Example analysis logic: 
result = "" 
        """

        if result_prompt not in code:
            code_prompt = result_prompt + "\n" + code
        else:
            code_prompt = code
        code_prompt = code_prompt.replace("```","# Code")
        print(code_prompt)
        ast.parse(code_prompt)  # Validate the code        
        local_vars = {}
        exec(code_prompt, {}, local_vars)  # Execute the code in a safe context
        # Return multiple output variables from the local_variables dictionary
        correlation_matrix = local_vars.get('correlation_matrix', None)
        pca_result = local_vars.get('pca_result', None)
        #dfCluster = local_vars.get('df', None)
        return correlation_matrix,pca_result#,dfCluster
    except Exception as e:
        raise RuntimeError(f"Error executing code: {e}")
    except SyntaxError as e:
        error_message = traceback.format_exc()
        print(error_message)
        corrected_code = query_llm_for_correction(code, error_message)
        if not corrected_code:
            fallback_on_failure(code, error_message)
            return  # Exit if fallback is reached
        return execute_code_with_tenacity(corrected_code)  # Retry with corrected code
    except Exception as e:
        print(f"Execution error: {e}")
        raise  # Trigger retry

def fallback_on_failure(code, error_message):
    """
    Log and skip the failed task as a fallback.
    """
    print(f"Fallback initiated for failed code:\n{code}")
    print(f"Error details: {error_message}")
    # Save details for later analysis
    with open("failed_code_log.txt", "a") as log_file:
        log_file.write(f"Code:\n{code}\nError:\n{error_message}\n\n")

def validate_and_execute_with_retries(code):
    global retry_count
    while retry_count < max_retries:
        try:
            validate_and_execute(code)
            break
        except Exception:
            retry_count += 1
            if retry_count >= max_retries:
                print("Max retries reached. Aborting.")

def validate_and_execute(code):
    """
    Validate and execute LLM-generated code. If there's a syntax error, request corrections.
    """
    try:
        # Validate syntax using ast
        ast.parse(code)
        print("Code is syntactically correct. Executing...")
        
        # Execute the code
        exec(code)
    except SyntaxError as e:
        print(f"Syntax error detected: {e}")
        print("Requesting LLM for a corrected version...")
        
        # Generate a prompt to fix the code
        error_message = traceback.format_exc()
        print(f"Error in Code{error_message}")
        corrected_code = query_llm_for_correction(code, error_message)
        
        if corrected_code:
            print("Retrying with corrected code...")
            validate_and_execute(corrected_code)
    except Exception as e:
        print(f"Error executing code: {e}")

def query_llm_for_correction(code, error_message):
    """
    Ask the LLM to correct the code based on the error message.
    """
    print(f"Error in Code{error_message}")
    prompt = (
        f"The following Python code contains a syntax error:\n\n"
        f"```\n{code}\n```\n\n"
        f"The error message is:\n\n"
        f"```\n{error_message}\n```\n\n"
        "Please provide a corrected version of the code."
    )
    
    corrected_code = query_llm(prompt)  # Replace with your LLM querying function
    return corrected_code

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
    for column in selected_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        file_path = os.path.join(output_path, f"distribution_{column}.png")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        #plt.legend(loc='upper right')  # Adding a legend for clarity
        plt.savefig(output_path)
        plt.savefig(file_path)
        plt.close()
           
def generate_dynamic_prompt(data_context,filename):
    """
    Creates a tailored prompt for the LLM based on dataset characteristics.

    Args:
        data_context (DataFrame): The dataset to analyze.

    Returns:
        str: A context-aware prompt for the LLM.
    """
    summary=gather_context(data_context,filename)
    # Extract the first 5 key-value pairs from the dictionary
    first_five_columns = list(summary['columns'].items())[:5]

    # Format the key-value pairs as a string
    formatted_columns = ', '.join([f"{key}: {value}" for key, value in first_five_columns])
    prompt = "You are a data analysis assistant.\n"
    prompt += (
            f"Name of the dataset is '{summary['filename']}' has Columns:{formatted_columns}.\n"
            #f"missing_values:{summary['missing_values']}"
            f"Summary statistics: {summary['summary']}\n"
            )
    numeric_cols = data_context.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data_context.select_dtypes(exclude=[np.number]).columns.tolist()

    # Highlight imbalances
    imbalance_warning = any(data_context[col].value_counts(normalize=True).max() > 0.9 for col in categorical_cols)
    if imbalance_warning:
        prompt += "- Note: One or more categorical columns are highly imbalanced.\n"

    # Add suggestions for numeric columns
    if numeric_cols:
        prompt += f"- Numeric columns: {numeric_cols}\n"
        prompt += "- Suggest advanced statistical analyses or feature engineering for numeric data  in not more than 150 words.\n"

    # Add suggestions for categorical columns
    if categorical_cols:
        prompt += f"- Categorical columns: {categorical_cols}\n"
        prompt += "- Suggest clustering or pattern recognition techniques for categorical data  in not more than 150 words.\n"
        
    print(f"Token count:{evaluate_prompt_efficiency(prompt)}")
    
    return prompt
def agentic_workflow(data):
    """
    Implements a multi-step workflow for data analysis using the LLM.

    Args:
        data (DataFrame): The dataset to analyze.

    Returns:
        None
    """
    numeric_data = data.select_dtypes(include=[np.number])
    # Step 1: Initial data exploration
    prompt = "Provide an exploratory analysis of this dataset  in not more than 150 words"
    prompt += f"Summary statistics: {data.describe()}"
    prompt += f"Correlation: {numeric_data.corr()}"
    insights = query_llm(prompt)

    # Step 2: Generate targeted insights
    follow_up_prompt = f"Based on your analysis: {insights}\nSuggest specific analyses or transformations in not more than 200 words."
    suggestions = query_llm(follow_up_prompt)

    # Execute suggestions and collect results
    analysis_results = []
    # Step 3: Execute suggestions iteratively
    for suggestion in suggestions.split('\n'):
        if len(suggestion)>0:
            result=execute_analysis_suggestion(data, suggestion)
            analysis_results.append(result)
    return analysis_results

def execute_analysis_suggestion(data, suggestion):
    """
    Executes a single suggestion from the LLM and returns the analysis summary.

    Args:
        data (DataFrame): The dataset to analyze.
        suggestion (str): Analysis suggestion to execute.

    Returns:
        str: A summary of the analysis performed.
    """
    if "outliers" in suggestion.lower():
        from sklearn.ensemble import IsolationForest
        numeric_data = data.select_dtypes(include=[np.number])
        outliers = IsolationForest().fit_predict(numeric_data)
        num_outliers = sum(outliers == -1)
        return f"Outlier Detection: Identified {num_outliers} outliers in the dataset."
    
    elif "pca" in suggestion.lower():
        from sklearn.decomposition import PCA
        numeric_data = data.select_dtypes(include=[np.number])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_data)
        explained_variance = pca.explained_variance_ratio_
        return (f"PCA Analysis: Reduced the dataset to 2 components. "
                f"Explained variance ratios are {explained_variance[0]:.2f} and {explained_variance[1]:.2f}.")
    
    elif "correlation" in suggestion.lower():
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        return "Correlation Analysis: Computed the correlation matrix for numeric features."
    
    else:
        return f"Analysis '{suggestion}' could not be executed or is not supported."

def evaluate_prompt_efficiency(prompt):
    """
    Evaluates the token usage of a given prompt.

    Args:
        prompt (str): The LLM prompt to evaluate.

    Returns:
        int: The number of tokens used.
    """
    # Assuming a tokenizer is available for the LLM being used
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = len(tokenizer.encode(prompt))
    return tokens

def validate_llm_code(code):
    """
    Validates the LLM-generated code by checking for restricted imports or risky operations.

    Args:
        code (str): The code string generated by the LLM.

    Returns:
        bool: True if the code is safe to execute, False otherwise.
    """
    restricted_keywords = ["os.system", "subprocess", "exec", "eval"]
    for keyword in restricted_keywords:
        if keyword in code:
            print(f"Potentially unsafe operation detected: {keyword}")
            return False
    return True
def create_analysis_folder(file_path):
    # Create a folder for storing analysis results, named after the dataset (without its file extension)
    # Extract the dataset name from the file path by removing the extension
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    charts_folder = dataset_name
    os.makedirs(charts_folder, exist_ok=True)
    return charts_folder

def impute_data(raw_data):
    #Handle missing data by imputing missing values for both numeric and categorical columns

    data_cleaned = raw_data.dropna()
    # Separate numeric and categorical columns
    numeric_data = data_cleaned.select_dtypes(include=[np.number])
    categorical_data = data_cleaned.select_dtypes(exclude=[np.number])

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
    return data_imputed

def do_correlation_analysis(data_imputed,charts_folder):
    # Generate and save a correlation heatmap for the imputed data, and extract top correlations
    heatmap_path = os.path.join(charts_folder, "correlation_heatmap.png")
    # Create a correlation heatmap and save it as a PNG image
    correlation_matrix = generate_correlation_heatmap(data_imputed, heatmap_path)
    
    # Unstack the correlation matrix and reset index for better readability
    correlations = correlation_matrix.unstack().reset_index()
    correlations.columns = ['Variable1', 'Variable2', 'Correlation']
    
    # Remove self-correlations (correlations of a variable with itself)
    correlations = correlations[correlations['Variable1'] != correlations['Variable2']]
    
    # Add an absolute correlation column for sorting
    correlations['AbsCorrelation'] = correlations['Correlation'].abs()
    
    # Sort correlations by absolute value and extract the top 3 correlations
    top_correlations = correlations.sort_values(by='AbsCorrelation', ascending=False).head(3)
    
def execute_LLM_code(code_response):
    # Validate and execute the code generated by a large language model (LLM)

    try:
        print("validating LLM-generated code...")
        valid_code = validate_llm_code(code_response)
        if valid_code:
            print("Executing LLM-generated code...")    
            exec(code_response)
        else:
            print("unsafe code..")
    except Exception as e:
        print(f"Error executing LLM-generated code: {e}")
def apply_LLM_analysis_generate_readme(data_imputed,file_path,charts_folder,best_columns):
    print("Gathering dataset context...")
    summary = gather_context(data_imputed, os.path.basename(file_path))
    
    # Ask the LLM to summarize the dataset
    print("Asking LLM for a summary of the dataset...")
    summary_response = interact_with_llm("summary", data_imputed,file_path)
    
    # Ask the LLM to suggest Python code for further analysis
    print("Asking LLM for Python code suggestions...")
    code_response = interact_with_llm("code", data_imputed,file_path)
    
    # Execute the suggested code (with caution)
    correlation_matrix,pca_result = execute_code_with_tenacity(code_response)
    
   
    code_exec_insight = f"Correlation Matrix:\n{correlation_matrix}\n"
    code_exec_insight += f"PCA variance:\n{pca_result}\n"
    #code_exec_insight += f"Kmeans Cluster:{cluster_summary}\n"
        
    # Ask the LLM for specific function calls
    print("Asking LLM for additional function call suggestions...")
    functions_response = interact_with_llm("function_call", data_imputed,file_path)
    
    #Implements a multi-step workflow for data analysis using the LLM. 
    analysis_results= agentic_workflow(data_imputed)
    
    #Generate image of the distribution
    charts = [os.path.join(charts_folder, f"distribution_{col}.png") for col in data_imputed.select_dtypes(include=[np.number]).columns if col in best_columns]
    print("Generating README.md...")
    
    generate_readme(data_imputed, summary, charts, summary_response,code_exec_insight,functions_response, analysis_results,charts_folder)

    print(f"Analysis complete. Results saved to README.md and visualizations saved in the '{charts_folder}' folder.")
    
def generate_readme(data, summary, charts, summary_response,code_exec_insight,functions_response,analysis_results,charts_folder):
    """
    Generates a README.md file summarizing the results of data analysis.

    Args:
        data (DataFrame): The dataset being analyzed.
        summary (dict): Summary statistics and metadata about the dataset.
        charts (list): List of file paths to generated visualizations.
        summary_response (str): Analytical insights provided by the LLM.
        code_response (str): Python code provided by the LLM.
        functions_response (str): Function API suggested provided by LLM.
        output_path (str): The path where the README file will be saved.

    Returns:
        None
    """
    output_path = os.path.join(charts_folder, "README.md")
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
        f.write(f"{summary_response}\n\n")
        f.write("\n## Analysis Results\n")
        for result in analysis_results:
            f.write(f"- {result}\n")
        
        #f.write("\n## Python Code suggested by LLM for further analysis\n")
        new_content = "\n## LLS Code Execution Results\n" + json.dumps(code_exec_insight, indent=4) + "\n"
        f.write(f"{new_content}\n\n")
        
        f.write("\n## Function API suggested by LLM for further analysis and insights\n")
        f.write(f"{functions_response}\n\n")

        f.write("## Visualizations\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")
def main():
    if len(sys.argv) != 2:
       print("Usage: uv run autolysis.py <dataset.csv>")
       sys.exit(1)

    load_dotenv()

    file_path = sys.argv[1]#'goodreads.csv' #sys.argv[1]
    raw_data = load_data(file_path)
    # Create a folder named after the dataset (without extension)
    charts_folder = create_analysis_folder(file_path)
    
    # Do Data cleansing
    data_imputed = impute_data(raw_data)    
 
    print("Selecting the best columns for visualization...")
    target_column = None  # Replace with the target column name if applicable
    best_columns = select_best_columns(data_imputed, target=target_column, max_columns=3)
    
    print("Generating visualizations...")
    do_correlation_analysis(data_imputed,charts_folder)
    
    generate_distribution_plots(data_imputed, charts_folder, best_columns)
    
    apply_LLM_analysis_generate_readme(data_imputed,file_path,charts_folder,best_columns)

if __name__ == "__main__":
    main()
