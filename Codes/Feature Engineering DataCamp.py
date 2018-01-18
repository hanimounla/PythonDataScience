import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# Figures inline and set visualization style
#matplotlib inline
sns.set()

# Import data
df = pd.read_csv('C:\Users\hani-_000\Documents\RStudio\\repos\Cleaning-Titanic-Data\\titanic_original.csv')

df_train, df_test = train_test_split(df, train_size = 0.7, random_state = 2)


# Store target variable of training data in a safe place
survived_train = df_train.survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['survived'], axis=1), df_test])

# View head
data.info()
data.name.tail()

# Extract Title from Name, store in column and plot barplot
data['Title'] = data.name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);