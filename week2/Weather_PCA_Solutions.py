# # Weather Data PCA Complete Solution
# ## Dimensionality Reduction Tutorial

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# %% [markdown]
# ## 1. Data Loading & Preparation

# %%
# Load dataset
weather = pd.read_csv('weatherAUS.csv')

# Initial inspection
print("Original shape:", weather.shape)
print("\nMissing values before cleaning:")
print(weather.isnull().mean().sort_values(ascending=False).head(10))

# %%
# Data cleaning
# Remove columns with >30% missing values
weather = weather.dropna(thresh=len(weather)*0.7, axis=1)

# Remove rows with remaining missing values
weather = weather.dropna()

print("\nShape after cleaning:", weather.shape)

# %% [markdown]
# ## 2. Feature Engineering

# %%
# Identify feature types
categorical = [var for var in weather.columns if weather[var].dtype=='O']
target = 'RainTomorrow'
continuous = [var for var in weather.columns if var not in categorical and var != target]

print(f"Categorical features ({len(categorical)}):\n{categorical}")
print(f"\nContinuous features ({len(continuous)}):\n{continuous}")

# %%
# Create new features
weather['HighHumidity'] = (weather['Humidity3pm'] > 75).astype(int)
weather['TempRange'] = weather['MaxTemp'] - weather['MinTemp']

# %% [markdown]
# ## 3. Data Visualization

# %%
# Categorical distribution
plt.figure(figsize=(10,4))
sns.countplot(x='RainToday', data=weather)
plt.title('Rain Today Distribution')
plt.show()

# %%
# Continuous distributions
plt.figure(figsize=(12,8))
for i, col in enumerate(['MaxTemp','Rainfall','WindSpeed9am','Humidity3pm']):
    plt.subplot(2,2,i+1)
    sns.histplot(weather[col], kde=True)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()

# ## 4. Principal Component Analysis

# %%
# Prepare data for PCA
X = weather[continuous]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# PCA with 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2'])
pca_df['RainTomorrow'] = weather['RainTomorrow']

# %%
# Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='RainTomorrow', 
    data=pca_df,
    alpha=0.6,
    palette=['skyblue','salmon']
)
plt.title('2D PCA Projection Colored by Rain Tomorrow')
plt.show()

# ## 5. Advanced PCA Analysis

# %%
# Scree plot
pca_full = PCA().fit(X_scaled)
plt.figure(figsize=(10,4))
plt.plot(range(1,11), pca_full.explained_variance_ratio_[:10], 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot (First 10 Components)')
plt.grid()
plt.show()

# %%
# Cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(10,4))
plt.plot(range(1,len(cumulative_variance)+1), cumulative_variance, 'o-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.grid()
plt.show()

print(f"Variance explained by 2 components: {cumulative_variance[1]:.2%}")
print(f"Variance explained by 3 components: {cumulative_variance[2]:.2%}")

# %%
# 3D PCA Visualization
pca_3d = PCA(n_components=3)
components_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Color by rain tomorrow
colors = ['skyblue' if x == 'No' else 'salmon' for x in weather['RainTomorrow']]
ax.scatter(
    components_3d[:,0], 
    components_3d[:,1], 
    components_3d[:,2],
    c=colors,
    alpha=0.5
)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('3D PCA Projection')
plt.show()
