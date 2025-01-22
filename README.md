import warnings
warnings.filterwarnings('ignore')
# Importing the numpy and pandas package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import scipy.stats as stats
import math
import itertools

from sklearn import metrics
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.neighbors import LocalOutlierFactor
import numpy as np

df=pd.read_csv("studentsPerformance.csv")
df.head(5)

df.tail(5)
num_columns = 3
num_rows = (len(df.columns) + num_columns - 1) // num_columns

# Create subplots
fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, num_rows * 5))
axes = axes.flatten()

# Iterate through all columns to visualize distributions
for i, column in enumerate(df.columns):
    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric variables: Plot histogram with KDE
        sns.histplot(df[column], kde=True, ax=axes[i], color='skyblue', bins=20)
        axes[i].set_title(f"Distribution of {column}", fontsize=14)
        axes[i].set_xlabel(column, fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)
    else:
        # Categorical variables: Plot bar chart
        sns.countplot(data=df, x=column, ax=axes[i], palette='pastel', order=df[column].value_counts().index)
        axes[i].set_title(f"Frequency of {column}", fontsize=14)
        axes[i].set_xlabel(column, fontsize=12)
        axes[i].set_ylabel("Count", fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

# Hide any unused subplots
for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.show()
