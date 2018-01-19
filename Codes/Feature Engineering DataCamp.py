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
df = pd.read_csv('DataSets\\titanicData.csv')

df_train, df_test = train_test_split(df, train_size = 0.70, random_state = 2)


# Store target variable of training data in a safe place
survived_train = df_train.survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['survived'], axis=1), df_test])

# View head
data.info()
data.name.tail()

# Extract Title from Name, store in column and plot barplot

def have_title(name):
    if re.search(' ([A-Z][a-z]+)\.',name):
        return re.findall(' ([A-Z][a-z]+)\.',name)
    else:
        return ['No_title']
    
titles_list = []
for full_name in data.name:
    name = str(full_name)
    titles_list.append(have_title(name))


    
    
titles = []
for title in titles_list:
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
data['has_cabin'] = ~data.cabin.isnull()

# View head of data
data.head()
data.info()

#len(data)-len(data.loc[(data['has_cabin'] == True)]) # passengers whom dont have cabine

# Drop columns and view head
data.drop(['cabin', 'name', 'ticket','home.dest','survived','boat','body'], axis=1, inplace=True)
data.head()

data.info()

data['age'] = data.age.fillna(data.age.median())
data['fare'] = data.fare.fillna(data.fare.median())
data['embarked'] = data['embarked'].fillna('S')

data.info()

# Binning numerical columns
data['catAge'] = pd.qcut(data.age, q=4, labels=False )
data['catFare']= pd.qcut(data.fare, q=4, labels=False)
data.head()

# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()

data = data.drop(['age', 'fare'], axis=1)
data.head()
data.reset_index()

# Create column of number of Family members onboard
data['fam_size'] = data.parch + data.sibsp

# Drop columns
data = data.drop(['sibsp','parch'], axis=1)
data.head()

# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()

##Creating Mchine Learning Module for dataset
# Split into test.train
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values

# Setup the hyperparameter grid
dep = np.arange(1,9)
param_grid = {'max_depth' : dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit it to the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))


Y_pred = clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('data/predictions/dec_tree_feat_eng.csv', index=False)




