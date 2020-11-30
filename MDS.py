import func
import numpy as np

def main(plot_corr=False):

    dataframe, X = func.read_file()

    if plot_corr: #choose to plot the heatmap or not
        func.plot_correlation(dataframe, 0.2)
    else:
        dataframe['type'] = dataframe['type'].astype(str)

        X_new = dataframe[["hair", "eggs", "milk", "aquatic", "toothed", "backbone", "breathes", "venomous", "tail", "catsize"]]
        X_new = np.array(X_new)
        X_new = func.compute_distance(X_new)

        components = func.compute_mds(X_new)

        func.scatter_plot(components, dataframe)

main() #run main(True) to plot the heatmap instead of the MDS plot