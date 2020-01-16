import pandas as pd
import numpy as np
from sklearn import tree, preprocessing, metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

#retrieve cotations of eurostoxx 50
data = pd.read_csv("C:/Users/franc/Documents/ESILV/A5/Trading_Platform/Eurostoxx50_EOD_Clean.csv", sep=";", decimal=",")
del data['Dates']
data = data[0:133]
returns = pd.DataFrame(index = range(len(data['ABI BB Equity']) - 1), columns = data.columns)
target = []

#compute daily returns
for col in data.columns:
    for i in range(len(data[col]) - 1):
        returns[col][i] = (float(data[col][i+1]) - float(data[col][i])) / float(data[col][i])
           
#center and normalize returns
normalizedReturns = preprocessing.scale(returns)
normalizedReturns = np.delete(normalizedReturns, 47, 1)

#compute target vector
for i in range(len(normalizedReturns)):
    if normalizedReturns[i, np.arange(47)].mean() > 0:
        target.append(1)
    elif normalizedReturns[i, np.arange(47)].mean() < 0:
        target.append(0)

#build decision tree, boosting classfier, or random forest classifier 
clf = tree.DecisionTreeClassifier()
boosting = GradientBoostingClassifier(learning_rate = 0.2)
randomForest = RandomForestClassifier(n_estimators = 20, max_depth = 2)

#build model and measure accuracy
def computeModel(model):
    X_train, X_test, y_train, y_test = train_test_split(normalizedReturns, target, test_size=0.5, random_state=42)
    model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_predict))

#plot importance of each feature
def getFeaturesImportances(model):
    feat_importances = pd.Series(model.feature_importances_, index=returns.iloc[:,0:47].columns)
    feat_importances.nlargest(10).plot(kind='barh')

#visualize graph of decision tree
def graphTree(model):
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = returns.iloc[:,0:47].columns, class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('stocks.png')
    Image(graph.create_png())

#choose between clf/boosting/randomForest
def main():
    computeModel(clf)
    getFeaturesImportances(clf)
    graphTree(clf)

if __name__ == "__main__":
    main()