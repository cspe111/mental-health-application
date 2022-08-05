import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('https://raw.githubusercontent.com/cspe111/mental-health-application/master/app/mentalHealthInTechSurvey_columnsCleaned.csv')

df['Gender'].replace(['Female', 'female', 'Trans-female', 'Cis Female', 'F', 'Woman', 'f',
                     'Femake', 'woman','Female ','cis-female/femme','Trans woman','Female (trans)',
                     'Female (cis)', 'femail'], 'Female', inplace=True)

df['Gender'].replace(['M','Male','male','m','Male-ish','maile','Cis Male','Mal','Male (CIS)',
                     'Make','Man','msle','Mail','cis male','Malr','Cis Man','Make','Male '], 'Male', inplace=True)

df['Gender'].replace(['something kinda male?', 'queer/she/they','non-binary','Nah','All','Enby',
                     'fluid','Genderqueer','Androgyne','Agender','Guy (-ish) ^_^',
                     'male leaning androgynous','Neuter','queer','A little about you','p',
                     'ostensibly male, unsure what that really means'], 'Other', inplace=True)

df.drop(df[df['Age'] < 0].index, inplace = True)
df.drop(df[df['Age'] > 100].index, inplace = True)

df['Interferes_Work'] = df['Interferes_Work'].fillna('Not sure')

# Label Encoder
categories =['Age', 'Gender', 'Family_History', 'Sought_Treatment', 'Interferes_Work',
             'Emp_Provide_Resources','Negative_Consequence', 'Discuss_w_Coworkers',
             'Discuss_w_Supervisor']
labelEncoder = preprocessing.LabelEncoder()

# Maps all the numerical values to each categorical label
mapping_dict = {}
for col in categories:
    df[col] = labelEncoder.fit_transform(df[col])
    name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col] = name_mapping

X = df.values[:, 0:8]
Y = df.values[:,8]

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)
decision_tree = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=4, min_samples_leaf=4)
decision_tree = decision_tree.fit(X_train, y_train)
y_prediction = decision_tree.predict(X_test)

# Serializes our model
import pickle
pickle.dump(decision_tree, open("model.pkl","wb"))