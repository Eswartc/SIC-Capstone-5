import numpy as np
import pandas as pd
import seaborn as sb
import sklearn.metrics as met
import sklearn.neighbors as ne
import matplotlib.pyplot as plt
import sklearn.linear_model as li
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import sklearn.model_selection as ms


#Section 1: Correlation between Numerical Features and label using histograms

""" np.random.seed(0)
plt.style.use('ggplot')

Nu = [[0, 'age'],
    [2, 'creatinine_phosphokinase'],
    [4, 'ejection_fraction'],
    [6, 'platelets'],
    [7, 'serum_creatinine'],
    [8, 'serum_sodium']]

Ca = [[1, 'anaemia'],
    [3, 'diabetes'],
    [5, 'high_blood_pressure'],
    [9, 'sex'],
    [10, 'smoking']]

NuNames = [i[1] for i in Nu]

DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
DF.drop(['time'], axis=1, inplace=True)

D = DF.to_numpy()

for i in Nu:
    X = D[:, i[0]]
    Y = D[:, -1]
    A = X[Y==0]
    B = X[Y==1]
    Bins = np.linspace(np.min(X), np.max(X), num=25)
    plt.hist([A, B], bins=Bins, color=['b', 'r'], label=['0', '1'])
    plt.xlabel(i[1])
    plt.ylabel('Frequency')
    plt.legend()
    plt.show() """


#Section 2: Correlation between Categorical Features and label using heatmaps

""" np.random.seed(0)
plt.style.use('ggplot')

Nu = [[0, 'age'],
    [2, 'creatinine_phosphokinase'],
    [4, 'ejection_fraction'],
    [6, 'platelets'],
    [7, 'serum_creatinine'],
    [8, 'serum_sodium']]

Ca = [[1, 'anaemia'],
    [3, 'diabetes'],
    [5, 'high_blood_pressure'],
    [9, 'sex'],
    [10, 'smoking']]

NuNames = [i[1] for i in Nu]
CaNames = [i[1] for i in Ca]

DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
DF.drop(['time'], axis=1, inplace=True)

D = DF.to_numpy()

for i in Ca:
    X = D[:, i[0]]
    Y = D[:, -1]
    M = np.zeros((2, 2))
    for x, y in zip(X, Y):
        M[int(x), int(y)] += 1
    sb.heatmap(M, annot=True, fmt='.0f',
            cmap='RdYlGn',
            xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('DEATH_EVENT')
    plt.ylabel(i[1])
    plt.show() """


#Section 3: KNN Model
#tr = training, te = testing, pr = precision, cr = classification report, ac = accuracy
""" def PrintReport(Model, trX, teX, trY, teY):
    trAc = Model.score(trX, trY)
    teAc = Model.score(teX, teY)
    trPr = Model.predict(trX)
    tePr = Model.predict(teX)
    trCR = met.classification_report(trY, trPr)
    teCR = met.classification_report(teY, tePr)
    print(f'{trAc = }')
    print(f'{teAc = }')
    print('_' * 50)
    print(f'Train CR:\n{trCR}')
    print('_' * 50)
    print(f'Test CR:\n{teCR}')
    print('_' * 50)

np.random.seed(0)
plt.style.use('ggplot')

DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
DF.drop(['time'], axis=1, inplace=True)

D = DF.to_numpy()

X = D[:, :-1]
Y = D[:, -1].reshape((-1, 1))

trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=5)

Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
trX = Scaler.fit_transform(trX0)
teX = Scaler.transform(teX0)

n0 = trY[trY == 0].size
n1 = trY[trY == 1].size

W = {0: n1/(n0+n1), 1: n0/(n0+n1)}
KNN = ne.KNeighborsClassifier(n_neighbors=5, weights='distance')
KNN.fit(trX, trY)
#PrintReport(KNN, trX, teX, trY, teY)         #to print report

def make_prediction(model, input_data):
    # Make predictions using the trained model
    predictions = model.predict(input_data)
    return predictions

# User input as a space-separated string
user_input_str = input("Enter 12 input values separated by spaces (e.g., 60 1 200 0 50 1 200000 1.2 140 1 0): ")

# Convert the input string to a list of numerical values
user_input_list = list(map(float, user_input_str.split()))

# Convert the list to a NumPy array
user_input = np.array([user_input_list])

# Make predictions
result = make_prediction(KNN, user_input)

# Display the prediction
print('Using KNN Model')
if result[0] == 1:
    print("Predicted Result: Yes")
else:
    print("Predicted Result: No") """



#Section 4: ANN Model
#tr = training, te = testing, pr = precision, cr = classification report, ac = accuracy

def PrintReport(Model, trX, teX, trY, teY):
    trAc = Model.score(trX, trY)
    teAc = Model.score(teX, teY)
    trPr = Model.predict(trX)
    tePr = Model.predict(teX)
    trCR = met.classification_report(trY, trPr)
    teCR = met.classification_report(teY, tePr)
    print(f'{trAc = }')
    print(f'{teAc = }')
    print('_' * 50)
    print(f'Train CR:\n{trCR}')
    print('_' * 50)
    print(f'Test CR:\n{teCR}')
    print('_' * 50)

np.random.seed(0)
plt.style.use('ggplot')

DF = pd.read_csv('data.csv', sep=',', header=0, encoding='utf-8')
DF.drop(['time'], axis=1, inplace=True)

D = DF.to_numpy()

X = D[:, :-1]
Y = D[:, -1].reshape((-1, 1))

trX0, teX0, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=5)

Scaler = pp.MinMaxScaler(feature_range=(-1, +1))
trX = Scaler.fit_transform(trX0)
teX = Scaler.transform(teX0)

n0 = trY[trY == 0].size
n1 = trY[trY == 1].size

W = {0: n1/(n0+n1), 1: n0/(n0+n1)}
MLP = nn.MLPClassifier(hidden_layer_sizes=(30), activation='relu', max_iter=10, random_state=0)
MLP.fit(trX, trY)

#PrintReport(MLP, trX, teX, trY, teY)  #to print report

def make_prediction(model, input_data):
    # Make predictions using the trained model
    predictions = model.predict(input_data)
    return predictions


# User input as a space-separated string
user_input_str = input("Enter 11 input values separated by spaces (e.g., 60 1 200 0 50 1 200000 1.2 140 1): ")

# Convert the input string to a list of numerical values
user_input_list = list(map(float, user_input_str.split()))

# Convert the list to a NumPy array
user_input = np.array([user_input_list])

# Make predictions for MLP model
result = make_prediction(MLP, user_input)

# Display the prediction
print('Using MLP Model')
if result[0] == 1:
    print("Predicted Result: Yes")
else:
    print("Predicted Result: No")


