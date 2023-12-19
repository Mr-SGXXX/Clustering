import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np

def gen_tsne(features:np.ndarray, true_labels, pred_labels, path, classIdx2label:callable=None):
    tsne = TSNE(n_components=2, random_state=2023)
    X_tsne = tsne.fit_transform(features)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1','Dim2'])  # 将降维后的特征转为DataFrame
    # pred_labels = self.dataset.classIdx2label(pred_labels)
    if true_labels is not None:
        if classIdx2label is not None:
            true_labels = classIdx2label(true_labels)  # 将真实标签从索引转换为实际标签
        df_tsne['label1'] = true_labels   # 将真实标签添加到DataFrame中
    df_tsne['label2'] = [chr(l + ord('a')) for l in pred_labels]  # 将预测标签添加到DataFrame中，并将其转换为字符格式
    df_tsne.head()
    plt.figure(figsize=(15, 9))     # 初始化绘图区域
    plt.title('TSNE Visualization of features')  # 设置标题
    if true_labels is not None:
        plt.subplot(1,2,1)    # 子图1
        sns.scatterplot(data=df_tsne, hue='label1', x='Dim1', y='Dim2')    # 绘制散点图
        plt.title('Ground Truth Label')   # 设置子图标题
        plt.legend(loc='best')   # 添加图例
        plt.subplot(1,2,2)    # 子图2
    sns.scatterplot(data=df_tsne, hue='label2', x='Dim1', y='Dim2')   # 绘制散点图
    plt.title('Predicted Label')   # 设置子图标题
    plt.legend(loc='best')   # 添加图例
    plt.savefig(path)    # 将图像保存在指定路径

def gen_umap(features:np.ndarray, true_labels, pred_labels, path, classIdx2label:callable=None):
    umap = UMAP(n_components=2, random_state=2023)     # 初始化UMAP算法
    X_umap = umap.fit_transform(features)    # 对输入特征进行UMAP降维
    df_umap = pd.DataFrame(X_umap, columns=['Dim1','Dim2'])  # 将降维后的特征转为DataFrame
    # pred_labels = self.dataset.classIdx2label(pred_labels)
    if true_labels is not None:
        if classIdx2label is not None:
            true_labels = classIdx2label(true_labels)  # 将真实标签从索引转换为实际标签
        df_umap['label1'] = true_labels   # 将真实标签添加到DataFrame中
    df_umap['label2'] = [chr(l + ord('a')) for l in pred_labels]  # 将预测标签添加到DataFrame中，并将其转换为字符格式
    df_umap.head()
    plt.figure(figsize=(15, 9))     # 初始化绘图区域
    plt.title('UMAP Visualization of features')  # 设置标题
    if true_labels is not None:
        plt.subplot(1,2,1)    # 子图1
        sns.scatterplot(data=df_umap, hue='label1', x='Dim1', y='Dim2')    # 绘制散点图
        plt.title('Ground Truth Label')   # 设置子图标题
        plt.legend(loc='best')   # 添加图例
        plt.subplot(1,2,2)    # 子图2
    sns.scatterplot(data=df_umap, hue='label2', x='Dim1', y='Dim2')   # 绘制散点图
    plt.title('Predicted Label')   # 设置子图标题
    plt.legend(loc='best')   # 添加图例
    plt.savefig(path)    # 将图像保存在指定路径

def gen_pretrain_loss_chart(loss_list, path, figsize=(10, 6)):
    epochs = list(range(1, len(loss_list) + 1))
    plt.figure(figsize=figsize)
    plt.plot(epochs, loss_list, color='steelblue', linewidth=2)
    plt.xlim(1, len(loss_list))
    plt.ylim(0, max(loss_list))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Pretrain Loss Chart", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.savefig(path)

def gen_clustering_chart_metrics_score(loss_list, score_dict, path, figsize=(10, 6)):
    epochs = list(range(1, len(loss_list) + 1))
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(epochs, loss_list, color='steelblue', linewidth=1)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax2 = ax1.twinx()
    if score_dict is not None:
        ax2.plot(epochs, score_dict['ACC'], color='red', linewidth=1, label='ACC')
        ax2.plot(epochs, score_dict['NMI'], color='orange', linewidth=1, label='NMI')
        ax2.plot(epochs, score_dict['ARI'], color='green', linewidth=1, label='ARI')
        ax2.set_ylabel("Evaluation Metric", color='black', fontsize=12)
        ax2.legend(loc='best')
        plt.title("Clustering Loss and Evaluation Metrics Chart", fontsize=14)
    else:
        plt.title("Clustering Loss Chart", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(path)

def gen_clustering_chart_clusters_score(loss_list, score_dict, path, figsize=(10, 6)):
    epochs = list(range(1, len(loss_list) + 1))
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(epochs, loss_list, color='steelblue', linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax2 = ax1.twinx()
    ax2.plot(epochs, score_dict['SC'], color='purple', linewidth=2, label='SC')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Clustering Loss and Clusters Metrics Chart", fontsize=14)
    plt.savefig(path)