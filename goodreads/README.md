# Automated Data Analysis

## Dataset Overview
Shape: (7860, 23)

### Columns and Data Types
- book_id: float64
- goodreads_book_id: float64
- best_book_id: float64
- work_id: float64
- books_count: float64
- isbn13: float64
- original_publication_year: float64
- average_rating: float64
- ratings_count: float64
- work_ratings_count: float64
- work_text_reviews_count: float64
- ratings_1: float64
- ratings_2: float64
- ratings_3: float64
- ratings_4: float64
- ratings_5: float64
- isbn: object
- authors: object
- original_title: object
- title: object
- language_code: object
- image_url: object
- small_image_url: object

### Missing Values
- book_id: 0 missing values
- goodreads_book_id: 0 missing values
- best_book_id: 0 missing values
- work_id: 0 missing values
- books_count: 0 missing values
- isbn13: 0 missing values
- original_publication_year: 0 missing values
- average_rating: 0 missing values
- ratings_count: 0 missing values
- work_ratings_count: 0 missing values
- work_text_reviews_count: 0 missing values
- ratings_1: 0 missing values
- ratings_2: 0 missing values
- ratings_3: 0 missing values
- ratings_4: 0 missing values
- ratings_5: 0 missing values
- isbn: 0 missing values
- authors: 0 missing values
- original_title: 0 missing values
- title: 0 missing values
- language_code: 0 missing values
- image_url: 0 missing values
- small_image_url: 0 missing values

## Insights from the LLM
1. **Feature Engineering**: Create interaction features such as the ratio of ratings in each category (e.g., `ratings_5` to `ratings_count`) to capture sentiment strength. Normalize features like `average_rating` and `work_text_reviews_count` to analyze patterns effectively.

2. **Advanced Statistical Analysis**: Use regression analysis to explore the relationship between `average_rating` and `book characteristics` (e.g., `books_count`, `ratings_count`) to predict high-rated books.

3. **Clustering Techniques**: Apply K-means clustering on `authors`, `original_title`, and `language_code` to identify groups of similar books, revealing patterns in genres or writing styles that attract higher ratings.


## Python Code suggested by LLM for further analysis
```python
import pandas as pd
import numpy as np
import chardet

# Detect encoding of the CSV file
with open('goodreads.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# Load the dataset with the detected encoding
df = pd.read_csv('goodreads.csv', encoding=encoding)

# Numeric columns for further analysis
numeric_columns = [
    'book_id', 'goodreads_book_id', 'best_book_id', 'work_id',
    'books_count', 'isbn13', 'original_publication_year', 
    'average_rating', 'ratings_count', 'work_ratings_count', 
    'work_text_reviews_count', 'ratings_1', 'ratings_2',
    'ratings_3', 'ratings_4', 'ratings_5'
]

# Exclude non-numeric columns for numeric analysis
numeric_df = df[numeric_columns]

# Advanced Statistical Analysis
# Correlation matrix
correlation_matrix = numeric_df.corr()

# Descriptive statistics
descriptive_stats = numeric_df.describe()

# Feature Engineering: Create a new feature for rating distribution
numeric_df['rating_distribution'] = (
    (numeric_df['ratings_1'] + 2 * numeric_df['ratings_2'] +
    3 * numeric_df['ratings_3'] + 4 * numeric_df['ratings_4'] +
    5 * numeric_df['ratings_5']) / numeric_df['ratings_count']
)

# Check for outliers in average_rating
Q1 = numeric_df['average_rating'].quantile(0.25)
Q3 = numeric_df['average_rating'].quantile(0.75)
IQR = Q3 - Q1
outliers = numeric_df[(numeric_df['average_rating'] < (Q1 - 1.5 * IQR)) | 
                      (numeric_df['average_rating'] > (Q3 + 1.5 * IQR))]

# Clustering techniques for categorical data
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Prepare categorical data for clustering
categorical_columns = [
    'isbn', 'authors', 'original_title', 'title', 
    'language_code', 'image_url', 'small_image_url'
]

# Encode categorical columns
encoded_df = df[categorical_columns].apply(LabelEncoder().fit_transform)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(encoded_df)
df['cluster'] = kmeans.labels_

# Display results
print("Correlation Matrix: \n", correlation_matrix)
print("\nDescriptive Statistics: \n", descriptive_stats)
print("\nOutliers in Average Rating: \n", outliers)
print("\nCluster Assignments: \n", df[['title', 'authors', 'cluster']].head())
```


## Function API suggested by LLM for further analysis and insights
To extract more insights from the `goodreads.csv` dataset, you can apply advanced statistical analyses, feature engineering techniques for the numeric data, and clustering or pattern recognition techniques for the categorical data. Here are some specific suggestions:

### Numeric Data Analysis

1. **Correlation Analysis**
    - To analyze relationships between numeric columns, you can compute the correlation matrix using:
      ```python
      import pandas as pd

      df = pd.read_csv('goodreads.csv')
      correlation_matrix = df.corr()
      print(correlation_matrix)
      ```

2. **Principal Component Analysis (PCA)**
    - PCA can help reduce dimensionality while preserving variance, which is useful for visualization.
      ```python
      from sklearn.decomposition import PCA
      from sklearn.preprocessing import StandardScaler

      # Assuming df_numeric only contains numeric columns
      df_numeric = df.select_dtypes(include=['float64'])
      scaler = StandardScaler()
      scaled_data = scaler.fit_transform(df_numeric)

      pca = PCA(n_components=2)
      pca_result = pca.fit_transform(scaled_data)
      print(pca_result)
      ```

3. **Feature Engineering**
    - Create new features based on existing ones. Examples include:
      - `rating_distribution`: Calculate the proportion of each rating.
      ```python
      df['rating_distribution'] = df[['ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']].div(df['ratings_count'], axis=0)
      ```

4. **Outlier Analysis**
    - Identify and handle outliers using boxplots or z-scores.
      ```python
      import seaborn as sns
      import matplotlib.pyplot as plt

      sns.boxplot(data=df['average_rating'])
      plt.show()
      ```

5. **Time Series Analysis**
    - Explore trends in the `original_publication_year` versus `average_rating` or `ratings_count`:
      ```python
      df['year'] = df['original_publication_year'].astype(int)
      yearly_avg_rating = df.groupby('year')['average_rating'].mean().reset_index()
      sns.lineplot(data=yearly_avg_rating, x='year', y='average_rating')
      plt.show()
      ```

### Categorical Data Analysis

1. **Clustering Authors**
    - Use K-means clustering to group authors based on their average ratings and number of books published:
      ```python
      from sklearn.cluster import KMeans

      author_stats = df.groupby('authors').agg({'average_rating': 'mean', 'books_count': 'sum'}).reset_index()
      kmeans = KMeans(n_clusters=3)
      author_stats['cluster'] = kmeans.fit_predict(author_stats[['average_rating', 'books_count']])
      ```

2. **Text Analysis**
    - Perform text analysis on `original_title` or `title` to identify common themes or topics using TF-IDF or Count Vectorization:
      ```python
      from sklearn.feature_extraction.text import TfidfVectorizer

      vectorizer = TfidfVectorizer(stop_words='english')
      X = vectorizer.fit_transform(df['title'])
      ```

3. **Analyzing Language Distribution**
    - Analyze the distribution of different `language_code`:
      ```python
      language_counts = df['language_code'].value_counts()
      language_counts.plot(kind='bar')
      plt.show()
      ```

4. **Pattern Recognition with Association Rules**
    - Use the Apriori algorithm to find associations between books, authors, or titles:
      ```python
      from mlxtend.frequent_patterns import apriori, association_rules
      
      frequent_items = apriori(df[['authors', 'title']], min_support=0.05, use_colnames=True)
      rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)
      print(rules)
      ```

5. **Visualizing Common Authors/Titles**
    - Use bar plots or word clouds to visualize the most common authors and titles.
      ```python
      sns.countplot(y='authors', data=df, order=df['authors'].value_counts().index)
      plt.show()
      ```

### Summary
These approaches will help you derive deeper insights and patterns from your `goodreads` dataset, enhancing your data analysis process. Each suggested method can be tailored based on specific needs and questions you want to address.

## Visualizations
![goodreads\distribution_best_book_id.png](goodreads\distribution_best_book_id.png)
![goodreads\distribution_work_id.png](goodreads\distribution_work_id.png)
![goodreads\distribution_isbn13.png](goodreads\distribution_isbn13.png)
