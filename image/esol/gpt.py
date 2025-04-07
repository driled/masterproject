
import os
import time
import numpy as np
import pandas as pd
from pub_func.table.data_loader import load_data, preprocess_data
from pub_func.table.dim_reduction import perform_pca, perform_umap, perform_autoencoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
  # 你已有的ILS模块
import pandas as pd
import numpy as np
from pub_func.table.clustering import evaluate_embedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pub_func.table.ils_clustering import ILS_clustering_with_kmeans_labels
from pub_func.table.visualization import (
    save_embedding_csv,
    plot_2d_embedding,
    plot_clustering_result,
    plot_silhouette_history,
    plot_comparison_bar,
    save_results_summary,
)




def run_kmeans_optimized_clustering(input_path, output_path, k_min=2, k_max=10):
    os.makedirs(output_path, exist_ok=True)

    # 加载和预处理数据
    X, _, _ = load_data(input_path)
    X_scaled, _ = preprocess_data(X)

    best_score = -1
    best_labels = None
    best_k = None
    silhouette_history = []

    print("正在尝试不同的聚类数以寻找最佳轮廓系数...")

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
            silhouette_history.append({'n_clusters': k, 'silhouette': score})
            print(f"k = {k}, silhouette = {score:.4f}")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        except Exception as e:
            print(f"k = {k} 聚类失败: {e}")

    # 保存聚类结果
    save_embedding_csv(X_scaled, output_path, "original_feature_space.csv")
    plot_clustering_result(
        X_scaled, best_labels, output_path,
        "original_kmeans_clustering.png",
        f"KMeans Clustering (k={best_k}) on Raw Features"
    )

    # 绘制轮廓系数趋势图
    plot_silhouette_history(
        silhouette_history, best_score, output_path,
        "original_kmeans_silhouette_history.png",
        "Silhouette Score vs Number of Clusters (KMeans)"
    )

    # 保存聚类分析摘要
    results_df = pd.DataFrame({
        "Method": ["KMeans"],
        "Parameters": [f"k={best_k}"],
        "Optimal Clusters": [best_k],
        "Silhouette Score": [best_score]
    })
    save_results_summary(results_df, output_path, "raw_kmeans_clustering_results.csv")

    print(f"完成！最佳簇数为 {best_k}，轮廓系数为 {best_score:.4f}")


def ILS(df, labelColumn, outColumn='LS', iterative=True):
    '''
    @author: amanda.parker@data61.csiro.au
    Citation and implemenatation details : ""

    Apply iterative label spreading in a multi-dimensional feature-space.
    Returns labels for all points and the order-labelled
    and distance-when-labelled for all newly labelled points.
    INPUTS :
        df = pandas dataFrame:
            all features are columns (and only those) +
            one column holding initial labels
        iterative = boolean :
            True : label spreading to unlabelled points applied iteratively
            False : all unlabelled points relabelled with regard to
                    initially labelled set
        labelColumn = String:
            Column name for column that holds initial labels.
            0 = to be labelled
            positive integers = assigned label.
    OUTPUTS :
        pandas dataSeries:
            index : same index input df
            name : outColumn
            data : Labels for all points
                (all values 0 replaced with a positive integer)
        pandas dataFrame:
            Only contains points that were labelled by ILS
            index : same as input df *reordered by order labelled*
            columns:
                minR : distance when relabelled
                IDclosestLabelled : ID of point label recieved from'
     '''

    featureColumns = [i for i in df.columns if i != labelColumn]
    # Keep original index columns in DF
    indexNames = df.index.names if any(df.index.names) else None

    oldIndex = df.index
    df = df.reset_index(drop=False)

    # separate labelled and unlabelled points
    labelled = [
        group for group in df.groupby(df[labelColumn] != 0)
    ][True][1].fillna(0)
    unlabelled = [
        group for group in df.groupby(df[labelColumn] != 0)
    ][False][1]

    # lists for ordered output data
    outD = []
    outID = []
    closeID = []

    # Continue while any point is unlabelled
    while len(unlabelled) > 0:
        # Calculate labelled to unlabelled distances matrix (D)
        D = pairwise_distances(
            labelled[featureColumns].values,
            unlabelled[featureColumns].values)

        # Find the minimum distance between a labelled and unlabelled point
        # first the argument in the D matrix
        (posL, posUnL) = np.unravel_index(D.argmin(), D.shape)
        # then convert to an index ID in the data frame
        # (The ordering will switch during iterations, more robust)
        idUnL = unlabelled.iloc[posUnL].name
        idL = labelled.iloc[posL].name

        # Switch label from 0 to new label
        unlabelled.loc[idUnL, labelColumn] = labelled.loc[idL, labelColumn]
        # move newly labelled point to labelled dataframe
        labelled = pd.concat([labelled, unlabelled.loc[[idUnL]]])

        # drop from unlabelled data frame
        unlabelled.drop(idUnL, inplace=True)

        # output the distance and id of the newly labelled point
        outD.append(D.min())
        outID.append(idUnL)
        closeID.append(idL)

    # Throw error if loose or duplicate points
    if len(labelled) + len(unlabelled) != len(df):
        raise Exception(
            '''The number of labelled ({}) and unlabelled ({}) 
                points does not sum to the total ({})'''.format(
                len(labelled), len(unlabelled), len(df)))

    # Reodered index for consistancy
    newIndex = oldIndex[outID]

    orderLabelled = pd.Series(
        data=outD, index=newIndex, name='minR')
    # ID of point label was spread from
    closest = pd.Series(
        data=closeID, index=newIndex, name='IDclosestLabel')
    labelled = labelled.rename(columns={labelColumn: outColumn})
    # new labels as dataseries
    newLabels = labelled[outColumn]

    # return
    return newLabels, pd.concat([orderLabelled, closest], axis=1)



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import os

def compute_kmeans_silhouette_for_embeddings(embeddings_dict, method_name, output_path, k_range=(2, 10)):
    """
    对降维嵌入执行 KMeans 聚类，搜索最佳簇数，并计算轮廓系数。

    参数：
        embeddings_dict: 降维数据字典，如 {'n_components=2': X_2, ...}
        method_name: 方法名（如 "PCA", "UMAP"）
        output_path: 结果保存路径
        k_range: 搜索簇数范围（默认2~10）

    返回：
        result_df: 记录最佳簇数和对应轮廓系数的DataFrame
    """
    results = []

    for param, X in embeddings_dict.items():
        print(f"\n[{method_name} {param}] 正在执行 KMeans 聚类...")

        best_score = -1
        best_k = None

        for k in range(k_range[0], k_range[1] + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                if len(np.unique(labels)) < 2:
                    score = np.nan
                else:
                    score = silhouette_score(X, labels)

                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"⚠️ k={k} 聚类失败: {e}")
                continue

        results.append({
            'Method': method_name,
            'Parameters': param,
            'Optimal Clusters': best_k,
            'Silhouette Score': best_score if best_score != -1 else np.nan
        })

    result_df = pd.DataFrame(results)
    os.makedirs(output_path, exist_ok=True)
    result_df.to_csv(os.path.join(output_path, f'{method_name.lower()}_kmeans_silhouette.csv'), index=False)

    print(f"\n✅ {method_name} 聚类分析完成，结果已保存。")
    return result_df




def ils_with_kmeans_on_embeddings(embeddings_dict, method_name, output_path, k_range=(2, 10)):
    """
    针对多个降维结果，使用 KMeans 初始化标签并执行 ILS 聚类，计算轮廓系数。

    参数：
        embeddings_dict: 降维数据字典，如 {'n_components=2': X2, ...}
        method_name: 方法名称（如 "PCA", "UMAP"）
        output_path: 保存路径
        k_range: 尝试的聚类簇数范围（默认 2~10）

    返回：
        result_df: 包含每种维度下最优簇数及对应轮廓系数的 DataFrame
    """
    os.makedirs(output_path, exist_ok=True)
    results = []

    for param, X in embeddings_dict.items():
        print(f"\n[{method_name} {param}] 执行 KMeans + ILS 聚类...")

        best_score = -1
        best_k = None
        best_labels = None

        for k in range(k_range[0], k_range[1] + 1):
            try:
                df = pd.DataFrame(X, columns=[f'dim_{i}' for i in range(X.shape[1])])
                df['label'] = 0

                # KMeans 聚类
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(df.drop(columns='label'))
                centroids = kmeans.cluster_centers_

                # 每个簇中选择一个中心点用于初始标记
                for i in range(k):
                    cluster_points = df[kmeans_labels == i]
                    if len(cluster_points) == 0:
                        continue
                    idx = min_toCentroid(cluster_points.drop(columns='label'), centroid=centroids[i])
                    df.loc[idx, 'label'] = i + 1  # ILS 中，0 表示未标记

                # ILS 标签传播
                new_labels_series, _ = ILS(df.copy(), labelColumn='label')
                final_labels = new_labels_series.values - 1

                # 轮廓系数计算
                if len(np.unique(final_labels)) < 2:
                    score = np.nan
                else:
                    score = silhouette_score(X, final_labels)

                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = final_labels

            except Exception as e:
                print(f"⚠️ k={k} 聚类失败：{e}")
                continue

        results.append({
            'Method': method_name,
            'Parameters': param,
            'Optimal Clusters': best_k,
            'Silhouette Score': best_score if best_score != -1 else np.nan
        })

    result_df = pd.DataFrame(results)
    save_path = os.path.join(output_path, f'{method_name.lower()}_ils_kmeans_results.csv')
    result_df.to_csv(save_path, index=False)
    print(f"\n✅ {method_name} ILS 聚类结果已保存：{save_path}")
    return result_df



def min_toCentroid(df, centroid=None, features=None):
    '''INPUT:
        df = pandas dataFrame:
                columns are dimensions
        centroid = list or tuple with consistant dimension
        features = string or list of strings:
                select only these columns of df
        '''

    if type(features) == type(None):
        features = df.columns

    if type(centroid) == type(None):
        centroid = df[features].mean()

    # distance from centroid for each point
    dist = df.apply(lambda row: sum(
        [(row[j] - centroid[i]) ** 2 for i, j in enumerate(features)]
    ), axis=1)

    # return index
    return dist.idxmin()



def ils_clustering_with_optimal_k(method_name, embeddings_dict, output_path, k_range=(2, 10)):
    """
    对降维嵌入的每个维度自动选择最优聚类数，执行 ILS 聚类并评估

    参数：
        method_name: 降维方法名（如 "PCA"）
        embeddings_dict: 降维结果字典，键为参数字符串，值为降维后数组
        output_path: 结果保存目录
        k_range: 尝试的簇数范围（默认从2到10）
    """
    os.makedirs(output_path, exist_ok=True)
    all_results = []
    silhouette_history_dict = {}

    for param, X in embeddings_dict.items():
        print(f"\n{method_name} - {param}: 正在搜索最优聚类数...")
        best_score = -1
        best_k = None
        best_labels = None
        best_new_labels = None
        silhouette_history = []

        for k in range(k_range[0], k_range[1] + 1):
            df = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(X.shape[1])])
            df['label'] = 0

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(df.drop(columns='label'))
            centroids = kmeans.cluster_centers_

            for i in range(k):
                cluster_points = df[kmeans_labels == i].copy()
                if len(cluster_points) == 0:
                    continue
                idx = min_toCentroid(cluster_points.drop(columns='label'), centroid=centroids[i])
                df.loc[idx, 'label'] = i + 1  # 0 表示未标记

            try:
                new_labels_series, _ = ILS(df.copy(), labelColumn='label')
                final_labels = new_labels_series.values - 1
                score = silhouette_score(X, final_labels)
                silhouette_history.append({'n_clusters': k, 'silhouette': score})

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = final_labels
                    best_new_labels = new_labels_series

            except Exception as e:
                print(f"k = {k} 聚类失败: {e}")
                continue

        # 保存最佳聚类结果
        all_results.append({
            'method': method_name,
            'params': param,
            'n_clusters': best_k,
            'silhouette_score': best_score
        })

        silhouette_history_dict[param] = silhouette_history

        # 结果可视化和导出
        save_embedding_csv(X, output_path, f"{method_name.lower()}_{param}.csv")

        if X.shape[1] == 2:
            plot_clustering_result(
                X, best_labels, output_path,
                f"{method_name.lower()}_{param}_ils_clusters.png",
                f"{method_name} {param} - ILS Clustering (k={best_k})"
            )

            plot_silhouette_history(
                silhouette_history, best_score, output_path,
                f"{method_name.lower()}_{param}_ils_silhouette.png",
                f"{method_name} {param} - Silhouette vs K"
            )

    # 保存CSV结果
    clustering_df = pd.DataFrame(all_results)
    save_results_summary(clustering_df, output_path, f"{method_name.lower()}_ils_clustering_results.csv")

    return clustering_df
from pub_func.table.dim_reduction import perform_pca, perform_umap, perform_autoencoder

def prepare_embeddings_dict(method, X_scaled, dims):
    """
    根据降维方法和指定维度列表，生成 embeddings_dict 字典

    参数：
        method: 降维方法名，'pca'、'umap' 或 'ae'
        X_scaled: 已标准化的特征数据（numpy array）
        dims: 降维维度列表，如 [2, 5, 10, 20]

    返回：
        embeddings_dict: { 'n_components=2': X_2d, ... }
    """
    embeddings_dict = {}

    for d in dims:
        if method == 'pca':
            X_reduced, _, _ = perform_pca(X_scaled, n_components=d)
        elif method == 'umap':
            X_reduced, _, _ = perform_umap(X_scaled, n_components=d, n_neighbors=15, min_dist=0.1)



        elif method == 'ae':
            X_reduced, _, _ = perform_autoencoder(X_scaled, latent_dim=d, intermediate_dim=50)

        else:
            raise ValueError(f"未知方法: {method}")

        embeddings_dict[f"n_components={d}"] = X_reduced

    return embeddings_dict



def compute_silhouette_for_embeddings(embeddings_dict, method_name, output_path, k_range=(2, 10)):
    """
    针对每个降维维度执行 ILS 聚类，自动搜索最优簇数，计算轮廓系数。

    参数：
        embeddings_dict: {'n_components=2': X2, 'n_components=5': X5, ...}
        method_name: 方法名称（PCA/UMAP/AE）
        output_path: 保存路径
        k_range: 尝试的聚类数量范围（默认 2~10）

    输出：
        DataFrame，其中包含每个降维维度的最佳 k 和对应轮廓系数
    """
    results = []

    for param, X in embeddings_dict.items():
        print(f"\n[{method_name} {param}] 正在计算轮廓系数...")

        best_score = -1
        best_k = None
        best_labels = None
        silhouette_history = []

        for k in range(k_range[0], k_range[1] + 1):
            df = pd.DataFrame(X, columns=[f'dim_{i}' for i in range(X.shape[1])])
            df['label'] = 0

            # KMeans 初始化
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(df.drop(columns='label'))
            centroids = kmeans.cluster_centers_

            # 每个簇找一个最近点作为初始标注
            for i in range(k):
                cluster_points = df[kmeans_labels == i]
                if len(cluster_points) == 0:
                    continue
                idx = min_toCentroid(cluster_points.drop(columns='label'), centroid=centroids[i])
                df.loc[idx, 'label'] = i + 1

            try:
                new_labels_series, _ = ILS(df.copy(), labelColumn='label')
                final_labels = new_labels_series.values - 1

                if len(np.unique(final_labels)) < 2:
                    score = np.nan
                else:
                    score = silhouette_score(X, final_labels)

                silhouette_history.append({'n_clusters': k, 'silhouette': score})

                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = final_labels

            except Exception as e:
                print(f"⚠️ k={k} 聚类失败：{e}")
                continue

        results.append({
            'Method': method_name,
            'Parameters': param,
            'Optimal Clusters': best_k,
            'Silhouette Score': best_score if best_score != -1 else np.nan
        })

    result_df = pd.DataFrame(results)
    os.makedirs(output_path, exist_ok=True)
    result_df.to_csv(os.path.join(output_path, f'{method_name.lower()}_ils_silhouette_results.csv'), index=False)

    print(f"\n{method_name} 所有维度的轮廓系数已保存到 CSV。")
    return result_df
# 假设你已有：
# pca_embeddings_dict = { 'n_components=2': X_pca_2, ... }









if __name__ == '__main__':
    input_path = r"D:\materproject\all-reps\ESOL\ESOL-table"
    output_path = r"D:\materproject\single-rep-rd\table"
    dims = [2, 5, 10, 20]
    X, _, _ = load_data(input_path)
    X_scaled, _ = preprocess_data(X)

    # # PCA
    # pca_embeddings_dict = prepare_embeddings_dict('pca', X_scaled, dims)
    #
    # # UMAP
    # umap_embeddings_dict = prepare_embeddings_dict('umap', X_scaled, [2, 5])
    #
    # # Autoencoder
    # ae_embeddings_dict = prepare_embeddings_dict('ae', X_scaled, dims)
    #
    # compute_silhouette_for_embeddings(pca_embeddings_dict, "PCA", output_path)
    # compute_silhouette_for_embeddings(umap_embeddings_dict, "UMAP", output_path)
    # compute_silhouette_for_embeddings(ae_embeddings_dict, "Autoencoder", output_path)
    #
    # pca_clustering_df = ils_clustering_with_optimal_k("PCA", pca_embeddings_dict, output_path)
    # umap_clustering_df = ils_clustering_with_optimal_k("UMAP", umap_embeddings_dict, output_path)
    # ae_clustering_df = ils_clustering_with_optimal_k("Autoencoder", ae_embeddings_dict, output_path)

    # PCA 降维结果字典
    pca_embeddings_dict = prepare_embeddings_dict('pca', X_scaled, dims)

    # UMAP 降维（常用 2 和 5）
    umap_embeddings_dict = prepare_embeddings_dict('umap', X_scaled, [2, 5])

    # AE 降维结果
    ae_embeddings_dict = prepare_embeddings_dict('ae', X_scaled, dims)

    # compute_kmeans_silhouette_for_embeddings(pca_embeddings_dict, "PCA", output_path)
    # compute_kmeans_silhouette_for_embeddings(umap_embeddings_dict, "UMAP", output_path)
    # compute_kmeans_silhouette_for_embeddings(ae_embeddings_dict, "Autoencoder", output_path)

    ils_with_kmeans_on_embeddings(pca_embeddings_dict, "PCA", output_path)
    ils_with_kmeans_on_embeddings(umap_embeddings_dict, "UMAP", output_path)
    ils_with_kmeans_on_embeddings(ae_embeddings_dict, "Autoencoder", output_path)



