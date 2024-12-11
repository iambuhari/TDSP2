# Automated Data Analysis

## Dataset Overview
Shape: (2363, 11)

### Columns and Data Types
- Country name: object
- year: int64
- Life Ladder: float64
- Log GDP per capita: float64
- Social support: float64
- Healthy life expectancy at birth: float64
- Freedom to make life choices: float64
- Generosity: float64
- Perceptions of corruption: float64
- Positive affect: float64
- Negative affect: float64

### Missing Values
- Country name: 0 missing values
- year: 0 missing values
- Life Ladder: 0 missing values
- Log GDP per capita: 28 missing values
- Social support: 13 missing values
- Healthy life expectancy at birth: 63 missing values
- Freedom to make life choices: 36 missing values
- Generosity: 81 missing values
- Perceptions of corruption: 125 missing values
- Positive affect: 24 missing values
- Negative affect: 16 missing values

## Insights from the LLM
Based on the dataset `happiness.csv`, here are the key insights:

### General Overview

- **Data Entries**: The dataset consists of a total of 2,363 entries covering various countries and years, from 2005 to 2023.
- **Unique Countries**: There are 165 unique countries represented, with Lebanon being the most frequently recorded country (18 occurrences).

### Summary Statistics

1. **Life Ladder**:
   - **Mean**: 5.48 (with a standard deviation of approximately 1.13), indicating the average perceived happiness level on a scale of 0 to 10.
   - **Range**: Values range from a minimum of 1.28 to a maximum of 8.02, showing significant variation in happiness across countries.

2. **Log GDP per Capita**:
   - **Mean**: 9.40 (standard deviation of about 1.15), which illustrates the average economic well-being per capita.
   - **Range**: The log GDP per capita varies from 5.53 to 11.68, indicating disparities in economic prosperity.

3. **Social Support**:
   - **Mean**: 0.81 (standard deviation of approximately 0.12), suggesting a generally high level of social support among the populations surveyed.
   - **Range**: Values span from 0.23 to 0.99, reflecting differences in the social structures of countries.

4. **Healthy Life Expectancy at Birth**:
   - **Mean**: 63.40 years (standard deviation of about 6.84), with a range from 6.72 years to 74.60 years.
   - This metric shows considerable variation and may correlate with health systems and quality of life.

5. **Freedom to Make Life Choices**:
   - **Mean**: 0.75 (standard deviation of about 0.14), implying that many people feel they have the freedom to make personal choices.
   - **Range**: Between 0.23 and 0.99, which showcases differing levels of perceived freedoms across nations.

6. **Generosity**:
   - **Mean**: 0.000098 (nearly zero), with a wide range (-0.34 to 0.70) indicating varying perceptions of generosity.
   - This could reflect cultural differences in charitable behavior or societal norms related to giving.

7. **Perceptions of Corruption**:
   - **Mean**: 0.74 (standard deviation of approximately 0.18), suggesting moderate to high perceptions of corruption among respondents across countries.
   - **Range**: From 0.035 to 0.98, indicating substantial differences in how corruption is perceived.

8. **Positive and Negative Affect**:
   - **Positive Affect Mean**: 0.65 (standard deviation of about 0.11), reflecting moderate positive feelings.
   - **Negative Affect Mean**: 0.27 (standard deviation of around 0.09), indicating lower levels of negative feelings on average.
   - The ranges for positive affect (0.18 to 0.88) and negative affect (0.08 to 0.71) show the emotional variability experienced by different populations.

### Conclusion

The analysis of the `happiness.csv` dataset reveals significant insights into the factors influencing happiness around the world, such as economic conditions (GDP), social structures (social support and freedom), health indicators (life expectancy), and emotional dimensions (positive and negative affects). The data indicates considerable disparities between different countries, demonstrating the complex interplay of various elements that contribute to well-being and life satisfaction.

## Visualizations
![happiness\correlation_heatmap.png](happiness\correlation_heatmap.png)
![happiness\distribution_year.png](happiness\distribution_year.png)
![happiness\distribution_Life Ladder.png](happiness\distribution_Life Ladder.png)
![happiness\distribution_Log GDP per capita.png](happiness\distribution_Log GDP per capita.png)
![happiness\distribution_Social support.png](happiness\distribution_Social support.png)
![happiness\distribution_Healthy life expectancy at birth.png](happiness\distribution_Healthy life expectancy at birth.png)
![happiness\distribution_Freedom to make life choices.png](happiness\distribution_Freedom to make life choices.png)
![happiness\distribution_Generosity.png](happiness\distribution_Generosity.png)
![happiness\distribution_Perceptions of corruption.png](happiness\distribution_Perceptions of corruption.png)
![happiness\distribution_Positive affect.png](happiness\distribution_Positive affect.png)
![happiness\distribution_Negative affect.png](happiness\distribution_Negative affect.png)
