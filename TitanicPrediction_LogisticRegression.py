import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import Imputer
from sklearn import linear_model as lm
from sklearn import tree
from sklearn.ensemble.forest import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB

class TitanicDisasterPrediction:
    
    def __init__(self,trainFilePath,testFilePath):
        '''
         Constructor for TitanicDisaster Prediction
        '''
        self.trainFilePath = trainFilePath
        self.testFilePath = testFilePath
        self.titanic_train_frame = pd.read_csv(self.trainFilePath, header=0)
        self.titanic_test_frame = pd.read_csv(self.testFilePath, header=0)
        
    
    def DataCleanup(self):
        '''
        Function to do DataCleanup of Titanic File. Fix the missing values
        '''
        print("Inside DataCleanup function")
        
        #Moving Survived as 1st column in Titanic_Train_Frame
        
        self.titanic_train_frame.insert(0, 'Survived_Column0', self.titanic_train_frame.Survived)
        
        
        # Defining new variable Gender based on sex
        
        self.titanic_train_frame['Gender'] = self.titanic_train_frame['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
        self.titanic_test_frame['Gender'] = self.titanic_test_frame['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
        
        #Finding Median Ages in each pclass variable
        median_ages = np.zeros((2,3)) 
        
        for i in range(0, 2):
            for j in range(0, 3):
                median_ages[i,j] = self.titanic_train_frame[(self.titanic_train_frame['Gender'] == i) & (self.titanic_train_frame['Pclass'] == j+1)]['Age'].dropna().median()
                median_ages[i,j] = self.titanic_test_frame[(self.titanic_test_frame['Gender'] == i) & (self.titanic_test_frame['Pclass'] == j+1)]['Age'].dropna().median()
        
        #Creating new Titanic_Train_Frame_AgeFill variable
        self.titanic_train_frame["AgeFill"] = self.titanic_train_frame["Age"]
        self.titanic_test_frame["AgeFill"] = self.titanic_test_frame["Age"]
        
        for i in range(0, 2):
            for j in range(0, 3):
                self.titanic_train_frame.loc[ (self.titanic_train_frame.Age.isnull()) & (self.titanic_train_frame.Gender == i) & (self.titanic_train_frame.Pclass == j+1),'AgeFill'] = median_ages[i,j]
                self.titanic_test_frame.loc[ (self.titanic_test_frame.Age.isnull()) & (self.titanic_test_frame.Gender == i) & (self.titanic_test_frame.Pclass == j+1),'AgeFill'] = median_ages[i,j]
        
        self.titanic_train_frame = self.titanic_train_frame.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Survived'], axis=1)
        self.titanic_test_frame = self.titanic_test_frame.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)  
        
   
    def FeatureEngineering(self):
        '''
        Function to convert categorical variables into dummy variables. 
        Create new variables based on feature engineering
        '''
        #Creating categorical variable for pclass and Fare
        pclass_frame  = pd.get_dummies(self.titanic_train_frame['Pclass'])
        del pclass_frame[3]
        self.titanic_train_frame = self.titanic_train_frame.join(pclass_frame)
        
        pclass_frame  = pd.get_dummies(self.titanic_test_frame['Pclass'])
        del pclass_frame[3]
        self.titanic_test_frame = self.titanic_test_frame.join(pclass_frame)
        
        #Creating categorical variable for Age*Class
        self.titanic_train_frame["Age*Class"] = self.titanic_train_frame["AgeFill"] * self.titanic_train_frame["Pclass"] 
        self.titanic_train_frame = self.titanic_train_frame.drop(['Age','Pclass'], axis=1) 
        
        self.titanic_test_frame["Age*Class"] = self.titanic_test_frame["AgeFill"] * self.titanic_test_frame["Pclass"] 
        self.titanic_test_frame = self.titanic_test_frame.drop(['Age','Pclass'], axis=1) 
           
        print(self.titanic_test_frame.describe)
        #print(self.titanic_test_frame)
        print("Inside Feature Engineering")
        
    def TitanicRegressionClassifier(self):
        '''
        Function to do Logistic Regression Classifer.
        '''
        train_Array = self.titanic_train_frame.values
        self.test_Array = self.titanic_test_frame.values
        #print(train_Array)
        
        logreg = lm.LogisticRegression(C=1e5)
        logreg.fit(train_Array[0::,1::],train_Array[0::,0])
        self.predicted_probability = logreg.predict(self.test_Array[0::,0::])
        self.predicted_probability_list = self.predicted_probability.tolist()
         
        print("Inside Regression classifer function")
    
    def TitanicDecisionTreeClassifier(self):
       
        '''
        Function to do DecisionTree Classifer.
        '''
        train_Array = self.titanic_train_frame.values
        self.test_Array = self.titanic_test_frame.values
        decTree = tree.DecisionTreeClassifier()
        decTree.fit(train_Array[0::,1::],train_Array[0::,0])
        self.predicted_probability = decTree.predict(self.test_Array[0::,0::])
        self.predicted_probability_list = self.predicted_probability.tolist()
        
    def RandomForestClassifer(self):
        
        '''
        Function to do RandomForest Classifer.
        '''
        train_Array = self.titanic_train_frame.values
        self.test_Array = self.titanic_test_frame.values
        randomForest = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
        randomForest.fit(train_Array[0::,1::],train_Array[0::,0])
        self.predicted_probability = randomForest.predict(self.test_Array[0::,0::])
        self.predicted_probability_list = self.predicted_probability.tolist()
        
    def NaiveBayesClassifer(self):
        
        '''
        Function to do Naive Bayes theorem based classifer. It is a conditional probability, application of Bayes Theorem P(A/B) = P(B/A)*P(A)/P(B)
        '''
        train_Array = self.titanic_train_frame.values
        self.test_Array = self.titanic_test_frame.values
        nb = GaussianNB()
        nb.fit(train_Array[0::,1::],train_Array[0::,0])
        self.predicted_probability = nb.predict(self.test_Array[0::,0::])
        self.predicted_probability_list = self.predicted_probability.tolist()
        
    def CreateSubmissionFile(self):
        '''
        Function to create submission file for kaggle submission
        '''
        # opening CSV file for writing
        trainwrite_file = open("C:/Hadoop/Dataset/Machine Learning/Titanic/submission.csv",'wt')
        trainWriter = csv.writer(trainwrite_file)
        rownum = 0
        self.test_Array = self.test_Array.tolist()
        
        while rownum < len(self.predicted_probability) :
            rowText = []
            passengerId =  self.test_Array[rownum][0]
            rowText.append(passengerId)
            rowText.append(self.predicted_probability_list[rownum])
            print(rowText)
            trainWriter.writerow(rowText)
            rownum = rownum + 1
    

predictor = TitanicDisasterPrediction("C:/Hadoop/Dataset/Machine Learning/Titanic/train.csv","C:/Hadoop/Dataset/Machine Learning/Titanic/test.csv")
predictor.DataCleanup()
predictor.FeatureEngineering()
#predictor.TitanicRegressionClassifier()
#predictor.TitanicDecisionTreeClassifier()
#predictor.RandomForestClassifer()
predictor.NaiveBayesClassifer()
predictor.CreateSubmissionFile()
    