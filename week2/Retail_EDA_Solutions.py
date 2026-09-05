# Black Friday EDA Tutorial Script

# ----------------------------------------
# 1. Introduction to EDA
# ----------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('BlackFriday.csv')

# Structure and statistics
print(df.info())
print(df.describe())

# Exercise 1.1: Initial Exploration

# 1. Missing values
print("Missing values per column:\n", df.isnull().sum())

# 2. Data types
print("Data types:\n", df.dtypes)

# ----------------------------------------
# 2. Statistical Analysis
# ----------------------------------------

# A. Central Tendency
print("Mean Purchase:", df['Purchase'].mean())
print("Median Purchase:", df['Purchase'].median())

# B. Spread & Distribution
print("IQR:", df['Purchase'].quantile(0.75) - df['Purchase'].quantile(0.25))
print("Skewness:", df['Purchase'].skew())

# Exercise 2.1: Advanced Statistics
print("90th percentile:", df['Purchase'].quantile(0.9))
print("Mean Purchase:", df['Purchase'].mean())
print("Median Purchase:", df['Purchase'].median())
# Interpretation: Right-skewed if mean > median

# ----------------------------------------
# 3. Data Visualization
# ----------------------------------------

# A. Distribution Plots
sns.histplot(df['Purchase'], bins=50, kde=True)
plt.title("Purchase Distribution")
plt.show()

# B. Outlier Detection
sns.boxplot(x='Purchase', data=df)
plt.title("Purchase Value Spread")
plt.show()

# Exercise 3.1: Visualization

# 1. Histogram for Product_Category_1
sns.histplot(df['Product_Category_1'], bins=30)
plt.title("Distribution of Product Category 1")
plt.show()

# 2. Boxplot of purchase by gender
sns.boxplot(x='Gender', y='Purchase', data=df)
plt.title("Purchase Amount by Gender")
plt.show()

# ----------------------------------------
# 4. Retail EDA Project
# ----------------------------------------

# Data Cleaning
clean_df = df.dropna(subset=['Product_Category_1', 'Purchase'])

# Optional: fill NaNs
clean_df['Product_Category_2'].fillna(-1, inplace=True)
clean_df['Product_Category_3'].fillna(-1, inplace=True)

# Remove outliers using IQR
Q1 = clean_df['Purchase'].quantile(0.25)
Q3 = clean_df['Purchase'].quantile(0.75)
IQR = Q3 - Q1
clean_df = clean_df[~((clean_df['Purchase'] < (Q1 - 1.5 * IQR)) | 
                      (clean_df['Purchase'] > (Q3 + 1.5 * IQR)))]

# Correlation Analysis
df_encoded = clean_df.copy()

# Encode all categorical columns
categorical_cols = df_encoded.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Correlation matrix
corr = df_encoded.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Encoded Features")
plt.show()

# Exercise 4.1: Advanced EDA

# 1. Top 3 products by average purchase
top_products= df.groupby('Product_ID')['Purchase'].mean().nlargest(3)
print("Top 3 Products with Highest Avg Purchase:\n", top_products)

# 2. Age vs Purchase by Gender
sns.scatterplot(x='Age', y='Purchase', hue='Gender', data=df, alpha=0.6)
plt.title("Age vs Purchase by Gender")
plt.xticks(rotation=45)
plt.show()