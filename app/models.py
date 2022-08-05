import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('https://raw.githubusercontent.com/cspe111/mental-health-application/master/app/mentalHealthInTechSurvey_columnsCleaned.csv')


df.drop(df[df['Age'] < 0].index, inplace=True)
df.drop(df[df['Age'] > 100].index, inplace=True)

df['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                      'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace=True)

df['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                      'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                      'woman', ], 'Female', inplace=True)

df["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                      'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman', ], 'Other', inplace=True)

df['work_interfere'] = df['work_interfere'].fillna('Not Sure')

# Label Encoder
categories =['Age', 'Gender', 'family_history', 'treatment', 'work_interfere', 'seek_help',
               'mental_health_consequence', 'coworkers', 'supervisor']
labelEncoder = preprocessing.LabelEncoder()

# Maps all the numerical values to each categorical label
mapping_dict = {}
for col in categories:
    df[col] = labelEncoder.fit_transform(df[col])
    le_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col] = le_mapping

X = df.values[:, 0:8]
Y = df.values[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
decision_tree = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
decision_tree.fit(X_train, y_train)
y_prediction = decision_tree.predict(X_test)

import pickle
pickle.dump(decision_tree, open("model.pkl", "wb"))