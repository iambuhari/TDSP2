# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "seaborn",
#   "matplotlib",
#   "openai",
#   "dotenv",
#   "requests",
#   "chardet",
#   "sklearn",
#   "statsmodels"
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

def detect_outliers(data):
    isolation_forest = IsolationForest(contamination=0.05)
    data['outliers'] = isolation_forest.fit_predict(data.select_dtypes(include=[float, int]))
    outliers = data[data['outliers'] == -1]
    return outliers

def perform_clustering(data):
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(data.select_dtypes(include=[float, int]))
    cluster_summary_text = (
    f"Cluster 0: {len(data[data['cluster'] == 0])} points, centered at {kmeans.cluster_centers_[0]}.\n"
    f"Cluster 1: {len(data[data['cluster'] == 1])} points, centered at {kmeans.cluster_centers_[1]}."
    )
    return cluster_summary_text

def perform_timeseries(data):
    decomposition = seasonal_decompose(data['time_series_column'], model='additive', period=12)
    decomposition.plot()

def perform_pca(data):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.select_dtypes(include=[float, int]))
    return pca
    
def generate_correlation_heatmap(data, output_path):
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()
    return correlation_matrix

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
    
    print("Performing basic analysis...")
    summary = basic_analysis(data_imputed)
    
    print("Performing outliers and anomaly detection...")
    outliers = detect_outliers(data_imputed)
    num_outliers = len(outliers)
    outlier_summary = f"Detected {num_outliers} outliers in dataset. These include values such as {outliers.iloc[:,0].tolist()}."
    
    print("Performing clustering to identify patterns...")
    cluster_summary_text = perform_clustering(data_imputed)
    
    
    print("Performing PCA analysis...")
    pca_results = perform_pca(data_imputed)
    # Explained variance
    explained_variance = pca_results.explained_variance_ratio_
    pca_summary = (
    f"PCA reduced the data to 2 components explaining {sum(explained_variance):.2f} of the variance.\n"
    f"Component 1: explains {explained_variance[0]:.2f} of the variance.\n"
    f"Component 2: explains {explained_variance[1]:.2f} of the variance."
)
    
    # Create a folder named after the dataset (without extension)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    charts_folder = dataset_name
    os.makedirs(charts_folder, exist_ok=True)
    
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

    # Convert to a readable summary
    corr_summary = [
        f"Correlation between {row['Variable1']} and {row['Variable2']} is {row['Correlation']:.2f}"
        for _, row in top_correlations.iterrows()
    ]
    corr_summary_text = "\n".join(corr_summary)
    
    generate_distribution_plots(data_imputed, charts_folder, best_columns)

    print("Consulting the LLM for insights...")
    llm_prompt = (
        f"You are a data analyst. Based on the following dataset analysis (summary statistics, correlations, clustering results), provide actionable insights and suggest what these insights might imply for a research application.\\n"
        f"Filename: {file_path}\\n"
        f"Columns and Data Types: {summary['columns']}\\n"           
        f"Sample data:{data.iloc[0]}\\n"
        f"Summary statistics: {summary['summary_statistics']}\\n"
        f"Correlation:{corr_summary_text}. Do you have any suggestions for further analysis or insights based on these correlations? \\n"
        f"Outliers:{outliers}. What insights can be derived from this? Could these be errors, anomalies, or opportunities? Suggest actions or further analyses? \\n"
        f"Clusters:{cluster_summary_text}. What insights can be derived from these clusters? How should we interpret the differences between them? Are there actionable opportunities or risks?\\n"
        f"PCA:{pca_summary}. What can be inferred from the variance distribution? Are there hidden patterns in the data we should explore further?\\n"
            
    )
    print(llm_prompt)
    llm_insights = query_llm(llm_prompt)

    print("Generating README.md...")
    charts = [heatmap_path] + [os.path.join(charts_folder, f"distribution_{col}.png") for col in data.select_dtypes(include=[np.number]).columns]
    generate_readme(data, summary, charts, llm_insights,charts_folder, "README.md")

    print(f"Analysis complete. Results saved to README.md and visualizations saved in the '{charts_folder}' folder.")
    
if __name__ == "__main__":
    main()
