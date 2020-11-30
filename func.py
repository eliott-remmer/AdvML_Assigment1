import pandas as pd
import numpy as np
import numpy.linalg as la

from sklearn.utils.graph import graph_shortest_path

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


def read_file():
    
    col_names = ['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed','backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']

    dataframe = pd.read_csv("/Users/georgasplund-sjunnesson/CodeProjects/AdvML/Assignment1/zoo.data", names=col_names)
    X = dataframe[col_names[1:-1]]
    return dataframe, X


def scatter_plot(components, dataframe):

    fig = px.scatter(components, x=0, y=1, color=dataframe['type'], text=dataframe['animal name'])
    fig.update_traces(marker=dict(size=24, line=dict(width=2, color='DarkSlateGrey')), textposition='bottom center', textfont_size=18)                          
    fig.update_layout(showlegend=False)
    fig.show()
    return


def plot_correlation(dataframe, target):

    plt.figure(figsize=(10,8))
    cor = dataframe.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

    #show corr with type
    cor_target = abs(cor["type"])
    
    #show best correlated variables
    relevant_features = cor_target[cor_target>target]
    print(relevant_features)
    plt.show()    
    return


def compute_distance(data):
    
    dist = np.array([[(np.sqrt(sum((x - y)**2))) for x in data] for y in data])
    return dist


def compute_mds(data):

    # centering the matrix
    n_samples = data.shape[0]
    rows_mean = np.sum(data, axis=0) / n_samples
    cols_mean = (np.sum(data, axis=1)/n_samples)[:, np.newaxis]
    all_mean = rows_mean.sum() / n_samples
    data = data - rows_mean - cols_mean + all_mean

    # obtain the eigen values and vectors
    eigen_val = la.eig(data)[0]
    eigen_vec = la.eig(data)[1].T

    # collect principal components
    PC1 = np.sqrt(np.abs(eigen_val[0]))*eigen_vec[0] 
    PC2 = np.sqrt(np.abs(eigen_val[1]))*eigen_vec[1]

    matrix_w = np.column_stack((PC1, PC2))
    return matrix_w.real


def compute_isomap(data, n_neighbors):
    
    dist = compute_distance(data)

    # saving the n nearest neighbors, setting the others to 0
    neighbors = np.zeros_like(dist)
    sort_dist = np.argsort(dist, axis=1)[:, 1:n_neighbors+1]
    for k,i in enumerate(sort_dist):
        neighbors[k,i] = dist[k,i]

    # get the shortest path using floyd warshall
    graph = -0.5 * (graph_shortest_path(neighbors, directed=False, method='FW') **2)

    # returns the mds on that graph
    return compute_mds(graph)