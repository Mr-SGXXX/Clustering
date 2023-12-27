import matplotlib.pyplot as plt
import torch
from umap import UMAP
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
import os

from metrics import Metrics
from utils import config

def draw_charts(rst_metrics:Metrics, pretrain_features:torch.Tensor, pretrain_loss_list:list, features:torch.Tensor, pred_labels, true_labels, description, cfg:config):
    figure_dir = os.path.join(cfg.get("global", "figure_dir"), description)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    method = cfg.get("global", "method_name")
    dataset = cfg.get("global", "dataset")
    figure_paths = []
    pretrain_tsne_figure_path = os.path.join(figure_dir, f"{method}_{dataset}_pretrain_tsne.png")
    pretrain_umap_figure_path = os.path.join(figure_dir, f"{method}_{dataset}_pretrain_umap.png")
    pretrain_loss_figure_path = os.path.join(figure_dir, f"{method}_{dataset}_pretrain_loss.png")
    clustering_tsne_figure_path = os.path.join(figure_dir, f"{method}_{dataset}_clustering_tsne.png")
    clustering_umap_figure_path = os.path.join(figure_dir, f"{method}_{dataset}_clustering_umap.png")
    clustering_loss_figure_path = os.path.join(figure_dir, f"{method}_{dataset}_clustering_loss.png")
    clustering_score_figure_path = os.path.join(figure_dir, f"{method}_{dataset}_clustering_score.png")
    if rst_metrics is not None:
        losses_list = []
        loss_names = []
        if len(rst_metrics.Loss) > 1:
            for loss_name in rst_metrics.Loss:
                loss_names.append(loss_name) 
                losses_list.append(rst_metrics.Loss[loss_name].val_list)
            gen_loss_chart(losses_list, loss_names, clustering_loss_figure_path)
            figure_paths.append(clustering_loss_figure_path)
        score_dict = {
            'ACC': rst_metrics.ACC.val_list,
            'NMI': rst_metrics.NMI.val_list,
            'ARI': rst_metrics.ARI.val_list,
            'SC': rst_metrics.SC.val_list,
        }
        gen_clustering_chart_metrics_score(rst_metrics.Loss['total_loss'].val_list, score_dict, clustering_score_figure_path)
        figure_paths.append(clustering_score_figure_path)
    if features is not None:
        gen_tsne(features.cpu().detach().numpy(), true_labels, pred_labels, clustering_tsne_figure_path)
        gen_umap(features.cpu().detach().numpy(), true_labels, pred_labels, clustering_umap_figure_path)
        figure_paths.append(clustering_tsne_figure_path)
        figure_paths.append(clustering_umap_figure_path)

    if pretrain_features is not None:
        gen_tsne(pretrain_features.cpu().detach().numpy(), true_labels, None, pretrain_tsne_figure_path)
        gen_umap(pretrain_features.cpu().detach().numpy(), true_labels, None, pretrain_umap_figure_path)
        figure_paths.append(pretrain_tsne_figure_path)
        figure_paths.append(pretrain_umap_figure_path)

    if pretrain_loss_list is not None and pretrain_loss_list != []:
        gen_pretrain_loss_chart(pretrain_loss_list, pretrain_loss_figure_path)
        figure_paths.append(pretrain_loss_figure_path)
    return figure_paths



def gen_tsne(features:np.ndarray, true_labels, pred_labels, path, classIdx2label:callable=None):
    # Generate t-SNE visualization of features
    tsne = TSNE(n_components=2, random_state=2023)
    X_tsne = tsne.fit_transform(features)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1','Dim2'])
    if true_labels is not None:
        if classIdx2label is not None:
            true_labels = classIdx2label(true_labels)
        df_tsne['label1'] = true_labels
    if pred_labels is not None:
        df_tsne['label2'] = [chr(l + ord('a')) for l in pred_labels]
    df_tsne.head()
    plt.figure(figsize=(15, 9))
    if true_labels is None and pred_labels is None:
        sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2')
        plt.legend(loc='best')
    elif true_labels is not None and pred_labels is None:
        sns.scatterplot(data=df_tsne, hue='label1', x='Dim1', y='Dim2')
        plt.title('Ground Truth Label TSNE')
        plt.legend(loc='best')
    elif true_labels is None and pred_labels is not None:
        sns.scatterplot(data=df_tsne, hue='label2', x='Dim1', y='Dim2')
        plt.title('Predicted Label TSNE')
        plt.legend(loc='best')
    else:
        plt.subplot(1,2,1)
        sns.scatterplot(data=df_tsne, hue='label1', x='Dim1', y='Dim2')
        plt.title('Ground Truth Label TSNE')
        plt.legend(loc='best')
        plt.subplot(1,2,2)
        sns.scatterplot(data=df_tsne, hue='label2', x='Dim1', y='Dim2')
        plt.title('Predicted Label TSNE')
        plt.legend(loc='best')
    plt.savefig(path)

def gen_umap(features:np.ndarray, true_labels, pred_labels, path, classIdx2label:callable=None):
    # Generate UMAP visualization of features
    umap = UMAP(n_components=2, random_state=2023)
    X_umap = umap.fit_transform(features)
    df_umap = pd.DataFrame(X_umap, columns=['Dim1','Dim2'])
    if true_labels is not None:
        if classIdx2label is not None:
            true_labels = classIdx2label(true_labels)
        df_umap['label1'] = true_labels
    if pred_labels is not None:
        df_umap['label2'] = [chr(l + ord('a')) for l in pred_labels]
    df_umap.head()
    plt.figure(figsize=(15, 9))
    if true_labels is None and pred_labels is None:
        sns.scatterplot(data=df_umap, x='Dim1', y='Dim2')
        plt.legend(loc='best')
    elif true_labels is not None and pred_labels is None:
        sns.scatterplot(data=df_umap, hue='label1', x='Dim1', y='Dim2')
        plt.title('Ground Truth Label UMAP')
        plt.legend(loc='best')
    elif true_labels is None and pred_labels is not None:
        sns.scatterplot(data=df_umap, hue='label2', x='Dim1', y='Dim2')
        plt.title('Predicted Label UMAP')
        plt.legend(loc='best')
    else:
        plt.subplot(1,2,1)
        sns.scatterplot(data=df_umap, hue='label1', x='Dim1', y='Dim2')
        plt.title('Ground Truth Label UMAP')
        plt.legend(loc='best')
        plt.subplot(1,2,2)
        sns.scatterplot(data=df_umap, hue='label2', x='Dim1', y='Dim2')
        plt.title('Predicted Label UMAP')
        plt.legend(loc='best')
    plt.savefig(path)

def gen_pretrain_loss_chart(loss_list, path, figsize=(10, 6)):
    # Generate pretrain loss chart
    epochs = list(range(1, len(loss_list) + 1))
    plt.figure(figsize=figsize)
    plt.plot(epochs, loss_list, color='steelblue', linewidth=2)
    plt.xlim(1, len(loss_list))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Pretrain Loss Chart", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.savefig(path)

def gen_loss_chart(losses_list, loss_names, path, figsize=(10, 6)):
    # Generate pretrain loss chart
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    epochs = list(range(1, len(losses_list[0]) + 1))
    plt.figure(figsize=figsize)
    for i in range(len(losses_list)):
        plt.plot(epochs, losses_list[i], color=colors[i], linewidth=2, label=loss_names[i])
    plt.xlim(1, len(losses_list[0]))
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Clustering Loss Chart", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.savefig(path)

def gen_clustering_chart_metrics_score(loss_list, score_dict, path, figsize=(10, 6)):
    # Generate clustering loss and evaluation metrics chart
    epochs = list(range(1, len(loss_list) + 1))
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(epochs, loss_list, color='steelblue', linewidth=1, label='totol loss')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax2 = ax1.twinx()
    if len(score_dict) > 1:
        ax2.plot(epochs, score_dict['ACC'], color='red', linewidth=1, label='ACC')
        ax2.plot(epochs, score_dict['NMI'], color='orange', linewidth=1, label='NMI')
        ax2.plot(epochs, score_dict['ARI'], color='green', linewidth=1, label='ARI')
        ax2.plot(epochs, score_dict['SC'], color='brown', linewidth=1, label='SC')
        ax2.set_ylabel("Evaluation Metric", color='black', fontsize=12)
    else:
        ax2.plot(epochs, score_dict['SC'], color='brown', linewidth=1, label='SC')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    plt.title("Clustering Loss and Evaluation Metrics Chart", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(path)