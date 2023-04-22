import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import warnings
import visualkeras
import time

from numpy import savetxt
from PIL import ImageFont

# machine learning
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, mean_squared_error
from scipy import stats
from scipy.stats import spearmanr

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Dropout

from tensorflow.keras import layers

from itertools import chain

class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)

        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights

        if(self.edge_weights is None):
            self.edge_weigths = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        self.preprocess = self.create_ffn(hidden_units, dropout_rate, name="preprocess")
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1"
        )
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2"
        )

        self.postprocess = self.create_ffn(hidden_units, dropout_rate, name="postprocess")
        self.compute_logits = tf.keras.layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_weights))
        x = x1 + x
        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x2 + x
        x = self.postprocess(x)
        node_embeddings = tf.squeeze(tf.gather(x, input_node_indices))
        return self.compute_logits(node_embeddings)
    

    def create_ffn(self, hidden_units, dropout_rate, name):
        ffn_layers = []
        for units in hidden_units[:-1]:
            ffn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

        ffn_layers.append(layers.Dense(units=hidden_units[-1]))
        ffn_layers.append(layers.Dropout(dropout_rate))

        ffn = keras.Sequential(ffn_layers)
        return ffn


def get_data(file_name):
    combination_data_path = os.path.join(os.path.dirname(__file__), 'data\\DrugCombinationData.tsv')
    df = pd.read_csv(combination_data_path, sep="\t")
    df["synergy_loewe"] = df["synergy_loewe"].round()
    df.loc[df["synergy_loewe"] >= 0, "synergy_loewe"] = 1
    df.loc[df["synergy_loewe"] < 0, "synergy_loewe"] = -1
    return df 

def convert(data):
    number = preprocessing.LabelEncoder()
    data = data.apply(number.fit_transform)
    return data

def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value

def spearman(y, pred):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value

def mse(y, pred):
    err = mean_squared_error(y, pred)
    print("Mean squared error is {}".format(err))
    return err

def squared_error(y,pred):
    errs = []
    for i in range(y.shape[0]):
        err = (y[i]-pred[i]) * (y[i]-pred[i])
        errs.append(err)
    return np.asarray(errs)

def logistic_regression(X_train, Y_train, X_test, Y_test, batch_size):
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    training_batch_generator = My_Custom_Generator(x_train, y_train, batch_size)
    validation_batch_generator = My_Custom_Generator(x_test, y_test, batch_size)

    for i in range(int((X_train.shape[0]/batch_size)+1)):
        batch_x, batch_y = training_batch_generator.__getitem__(i)
        logreg.fit(batch_x, batch_y)
    for i in range(int((X_test.shape[0]/batch_size)+1)):
        batch_x, batch_y = validation_batch_generator.__getitem__(i)
        Y_pred = logreg.predict(batch_x)     

    acc_logreg = round(logreg.score(X_train, Y_train) * 100, 2)
    print(acc_logreg)

    mse_value = mse(Y_test, Y_pred)
    spearman_value = spearman(Y_test, Y_pred)
    pearson_value = pearson(Y_test, Y_pred)    

def random_forest(X_train, Y_train, X_test, Y_test, batch_size):
    random_forest = RandomForestClassifier(n_estimators=100)
    training_batch_generator = My_Custom_Generator(x_train, y_train, batch_size)
    validation_batch_generator = My_Custom_Generator(x_test, y_test, batch_size)

    for i in range(int((X_train.shape[0]/batch_size)+1)):
        batch_x, batch_y = training_batch_generator.__getitem__(i)
        random_forest.fit(batch_x, batch_y)

    Y_pred = []
    for i in range(int((X_test.shape[0]/batch_size)+1)):
        batch_x, batch_y = validation_batch_generator.__getitem__(i)
        Y_pred.append(random_forest.predict(batch_x))

    Y_pred = list(chain.from_iterable(Y_pred))
    Y_pred = np.array(Y_pred)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)

    #savetxt('data.csv', Y_test, delimiter=',')
    #savetxt('data1.csv', Y_pred, delimiter=',')
    mse_value = mse(Y_test, Y_pred)
    spearman_value = spearman(Y_test, Y_pred)
    pearson_value = pearson(Y_test, Y_pred)

def NeuralModel(X_train, Y_train, X_test, Y_test, batch_size):
    model = Sequential()
    model.add(Dense(12, input_shape=(3,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    print("Processing...")
    
    font = ImageFont.truetype("arial.ttf", 32)
    #visualkeras.layered_view(model).show()

    model.fit(x_train, y_train, epochs=20, batch_size=batch_size, verbose=0)
    time.sleep(86400)
    Y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    Y_pred.flatten()
    _, accuracy = model.evaluate(X_train, Y_train)
    print('Accuracy: %.2f' % (accuracy*100)) 

    mse_value = mse(Y_test, Y_pred)
    spearman_value = spearman(Y_test, Y_pred)
    pearson_value = pearson(Y_test, Y_pred)

def GraphModel(data, x_train, y_train, x_test, y_test, batch_size):
    node_features = data.iloc[:, 0:2]
    edges = data.iloc[:, 3]
    edge_weights = tf.ones(shape=edges.shape[0])
    graph_info = (node_features, edges, edge_weights)
    hidden_units = [32, 32]
    learning_rate = 0.01
    dropout_rate = 0.5
    num_epochs = 300
    batch_size = 256
    num_classes = 1
    gnn_model = GNNNodeClassifier(
        graph_info=graph_info,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        name="gnn_model",
    )
    gnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = gnn_model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )
    Y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    Y_pred.flatten()
    _, accuracy = gnn_model.evaluate(X_train, Y_train)

def GetMetrics():
    input_transform = Percentage()
    relative_in_transform = Percentage()
    relative_out_transform = None
    return
    [
        MeanSquaredErrorWrapper(y_true_transformer=input_transform,
                                y_pred_transformer=None),
        MeanAbsoluteErrorWrapper(y_true_transformer=input_transform,
                                    y_pred_transformer=None),
        MeanAbsolutePercentageErrorWrapper(y_true_transformer=relative_in_transform,
                                            y_pred_transformer=relative_out_transform),
        BrayCurtisDissimilarity(y_true_transformer=relative_in_transform,
                                y_pred_transformer=relative_out_transform),
        PearsonCorrelation(y_true_transformer=relative_in_transform,
                            y_pred_transformer=relative_out_transform),
        SpearmanCorrelation(y_true_transformer=relative_in_transform,
                            y_pred_transformer=relative_out_transform),
        JensenShannonDivergence(y_true_transformer=relative_in_transform,
                                y_pred_transformer=relative_out_transform),
        # CrossEntropy(y_true_transformer=relative_in_transform,
        #             y_pred_transformer=relative_out_transform),
    ]    

warnings.filterwarnings("ignore")
dataPath = os.path.join(os.path.dirname(__file__), 'data\\summary_v_1_5_Bone.xlsx')
data = pd.read_excel(dataPath, usecols="B, C, D, T")
data['drug_row'] = data['drug_row'].astype('string')
data['drug_col'] = data['drug_col'].astype('string')
data['cell_line_name'] = data['cell_line_name'].astype('string') 
#training_drug_info = convert(data)
training_drug_info = convert(get_data('data/DrugCombinationData.tsv'))
training_drug_info = training_drug_info.sample(n=200000)
x_train = training_drug_info.drop("synergy_loewe", axis=1)
y_train = training_drug_info["synergy_loewe"]
batch_size = 516
print(x_train.shape)
print(y_train.shape)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=100)#, stratify=y_train)
#GraphModel(training_drug_info, x_train, y_train, x_test, y_test, batch_size)
print("Neural:")
NeuralModel(x_train, y_train, x_test, y_test, batch_size)
print("RandomForest:")
random_forest(x_train, y_train, x_test, y_test, batch_size)
print("LogisitcReg:")
logistic_regression(x_train, y_train, x_test, y_test, batch_size)


