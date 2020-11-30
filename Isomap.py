import func
import numpy as np

def main():
    dataframe, X = func.read_file()
    dataframe['type'] = dataframe['type'].astype(str)

    X = np.array(X)
    components = func.compute_isomap(X,n_neighbors=6)

    func.scatter_plot(components, dataframe)

main()