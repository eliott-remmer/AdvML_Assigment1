import func
from sklearn.decomposition import PCA

def main():
    dataframe, X = func.read_file()
    dataframe['type'] = dataframe['type'].astype(str)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    func.scatter_plot(components, dataframe)

main()