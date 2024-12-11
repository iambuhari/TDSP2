# Automated Data Analysis

## Dataset Overview
Shape: (10000, 23)

### Columns and Data Types
- book_id: int64
- goodreads_book_id: int64
- best_book_id: int64
- work_id: int64
- books_count: int64
- isbn: object
- isbn13: float64
- authors: object
- original_publication_year: float64
- original_title: object
- title: object
- language_code: object
- average_rating: float64
- ratings_count: int64
- work_ratings_count: int64
- work_text_reviews_count: int64
- ratings_1: int64
- ratings_2: int64
- ratings_3: int64
- ratings_4: int64
- ratings_5: int64
- image_url: object
- small_image_url: object

### Missing Values
- book_id: 0 missing values
- goodreads_book_id: 0 missing values
- best_book_id: 0 missing values
- work_id: 0 missing values
- books_count: 0 missing values
- isbn: 700 missing values
- isbn13: 585 missing values
- authors: 0 missing values
- original_publication_year: 21 missing values
- original_title: 585 missing values
- title: 0 missing values
- language_code: 1084 missing values
- average_rating: 0 missing values
- ratings_count: 0 missing values
- work_ratings_count: 0 missing values
- work_text_reviews_count: 0 missing values
- ratings_1: 0 missing values
- ratings_2: 0 missing values
- ratings_3: 0 missing values
- ratings_4: 0 missing values
- ratings_5: 0 missing values
- image_url: 0 missing values
- small_image_url: 0 missing values

## Insights from the LLM
Here is a summary of insights from the `goodreads.csv` dataset:

### General Overview:
- The dataset contains information on 10,000 books from Goodreads.
- Key columns include identifiers, metadata about books, authors, publication year, language, ratings, and counts of reviews.

### Key Statistics:
- **Books Count**: On average, each entry corresponds to approximately 75.71 books, with the maximum being 3455. 
- **Authors**: There are 4664 unique authors, with Stephen King being the most frequently mentioned (60 times).
- **Publication Year**: The average original publication year is approximately 1982, with entries spanning from as early as 1750 to 2017.

### Rating Insights:
- **Average Rating**: The average rating for the books is 4.00, with a standard deviation of approximately 0.25, indicating good overall reception among the books.
- **Ratings Count**: Books vary widely in their total number of ratings. The maximum is 4,780,653 with a mean of approximately 54,001 ratings.
- **Distribution of Ratings**: 
  - 5-Star Ratings: Mean count of about 23,790.
  - 4-Star Ratings: Mean count of about 19,966.
  - 1-Star Ratings: Mean count of about 1,345.
  - Most books receive a significant majority of higher ratings, illustrating a generally favorable perception.

### Language Distribution:
- **Language Codes**: The dataset shows entries primarily in English ('eng') which accounts for 6341 entries, and a total of 25 unique language codes present.

### Publication and ISBN Information:
- **ISBN Information**: Out of the total entries, 9,300 books have a valid ISBN (ISBN-10), and 9,415 have an ISBN-13.
- **ISBN-13**: The numeric values have a mean around 9.75 trillion, reflecting the use of the ISBN-13 standard.

### Visual Representations:
- **Cover Images**: There are 6,669 unique cover images, with many books sharing the same image URL, indicating multiple editions or formats of certain titles.

### Conclusion:
This dataset provides a rich source of information for analysis regarding the popularity, distribution, and characteristics of books on Goodreads. The analysis can be further expanded to explore trends in ratings over time, correlations between the number of ratings and the average ratings, or the influence of different authors and their works. Additionally, insights into reader preferences based on language and publication year can enhance understanding of the literary market.

## Visualizations
![goodreads\correlation_heatmap.png](goodreads\correlation_heatmap.png)
![goodreads\distribution_book_id.png](goodreads\distribution_book_id.png)
![goodreads\distribution_goodreads_book_id.png](goodreads\distribution_goodreads_book_id.png)
![goodreads\distribution_best_book_id.png](goodreads\distribution_best_book_id.png)
![goodreads\distribution_work_id.png](goodreads\distribution_work_id.png)
![goodreads\distribution_books_count.png](goodreads\distribution_books_count.png)
![goodreads\distribution_isbn13.png](goodreads\distribution_isbn13.png)
![goodreads\distribution_original_publication_year.png](goodreads\distribution_original_publication_year.png)
![goodreads\distribution_average_rating.png](goodreads\distribution_average_rating.png)
![goodreads\distribution_ratings_count.png](goodreads\distribution_ratings_count.png)
![goodreads\distribution_work_ratings_count.png](goodreads\distribution_work_ratings_count.png)
![goodreads\distribution_work_text_reviews_count.png](goodreads\distribution_work_text_reviews_count.png)
![goodreads\distribution_ratings_1.png](goodreads\distribution_ratings_1.png)
![goodreads\distribution_ratings_2.png](goodreads\distribution_ratings_2.png)
![goodreads\distribution_ratings_3.png](goodreads\distribution_ratings_3.png)
![goodreads\distribution_ratings_4.png](goodreads\distribution_ratings_4.png)
![goodreads\distribution_ratings_5.png](goodreads\distribution_ratings_5.png)
