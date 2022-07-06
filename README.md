# FIMClusters

The repository contains a script which takes tabular data of FIM scores of stroke patients across multiple weeks.

### Structure
The script *FIM.py* contains the main class *FIMData*, which loads the data and performs the clustering. An example usage is:
```python
fim = FIMData(file_name="data.xlsx", sheet_name="FIM", save_folder="Results")
fim.compute_delta()
out, probs = fim.outliers_detection()
if remove_outliers:
    fim.remove_outliers()
fim.select_clusters()
fim.clustering(n_clusters=3)
```

The class parameters are the name of the Excel file with data and the sheet name, together with an optional folder where to save the results. `compute_delta()` computes the delta feature between consecutive weeks. `outliers_detection()` applies the Mahalanobis distance to check for possible outliers (important for later clustering, as if some outliers remain, they may form their own cluster); the outliers are removed from the dataset with `remove_outliers()` method. The `select_clusters()` method computes metrics of clusters goodness for different values of clusters and plots it. Finally, the `clustering(n_clusters=3)` method performs the actual clustering with the specified number of clusters, saving the cluster labels and the distances from each cluster center in two files.
