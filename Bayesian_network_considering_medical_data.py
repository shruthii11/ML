import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
data = pd.read_csv('Prog8_Bayesian_network.csv')
model = BayesianNetwork()
model.add_edges_from([('age', 'heartDisease'), 
                      ('sex', 'heartDisease'), 
                      ('cp', 'heartDisease'), 
                      ('trestbps', 'heartDisease'), 
                      ('chol', 'heartDisease')])
model.fit(data, estimator=MaximumLikelihoodEstimator)
age = input("Enter age (SeniorCitizen/Teen/Youth/MiddleAged): ")
sex = input("Enter sex (Male/Female): ")
cp = input("Enter chest pain type (Typical angina/Atypical angina/Non-anginal pain): ")
trestbps = input("Enter resting blood pressure (High/Normal/Low): ")
chol = input("Enter cholesterol level (High/Normal/Low): ")
user_data = pd.DataFrame({'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol]})
prediction =  model.predict(user_data)
print("Prediction:", prediction)