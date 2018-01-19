import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# Figures inline and set visualization style
%matplotlib inline
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
titles_list = []
for full_name in data.name:
    name = str(full_name)
    titles_list.append(have_title(name))


def have_title(name):
    if re.search(' ([A-Z][a-z]+)\.',name):
        return re.findall(' ([A-Z][a-z]+)\.',name)
    else:
        return ['No_title']
    
    
titles = []
for title in titles_list:
    print title
    titles.append(title[0])

data['title'] = titles
#data['Title'] = data.name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.',x).group(1)) #not working !
sns.countplot(x='title', data=data);
plt.xticks(rotation=45);

#Regulize titles
data['title'] = data['title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['title'] = data['title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'No_title'],'Special')
sns.countplot(x='title', data=data);
plt.xticks(rotation=45);

# Did they have a Cabin?
data['Has_Cabin'] = ~data.cabin.isnull()

# View head of data
data.head()

len(data)-len(data.loc[(data['Has_Cabin'] == True)])







