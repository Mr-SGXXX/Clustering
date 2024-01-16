# Copyright (c) 2023 Yuxuan Shao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import matplotlib.pyplot as plt
import torch
from umap import UMAP
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
import typing
import os

from metrics import Metrics
from utils import config


def draw_charts(rst_metrics: typing.Union[Metrics, None],
                pretrain_features: typing.Union[torch.Tensor, np.ndarray, None],
                features: typing.Union[torch.Tensor, np.ndarray, None],
                pred_labels: np.ndarray, true_labels: typing.Union[np.ndarray, None], description: str, cfg: config):
    """
    Generate figures for clustering results

    Args:
        rst_metrics (typing.Union[Metrics, None]): The metrics of the clustering results, None if not available
        pretrain_features (typing.Union[torch.Tensor, np.ndarray, None]): The features before clustering, None if not available
        features (typing.Union[torch.Tensor, np.ndarray, None]): The features after clustering, None if not available
        pred_labels (np.ndarray): The predicted labels
        true_labels (typing.Union[np.ndarray, None]): The ground truth labels, None if not available
        description (str): The description of the experiment
        cfg (config): The config of the experiment
    Returns:
        figure_paths (list): The paths of the generated figures
    """
    method = cfg.get("global", "method_name")
    dataset = cfg.get("global", "dataset")
    figure_dir = os.path.join(
        cfg.get("global", "figure_dir"), method, dataset, description)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    figure_paths = []
    pretrain_tsne_figure_path = os.path.join(
        figure_dir, f"{method}_{dataset}_pretrain_tsne.png")
    pretrain_umap_figure_path = os.path.join(
        figure_dir, f"{method}_{dataset}_pretrain_umap.png")
    pretrain_loss_figure_path = os.path.join(
        figure_dir, f"{method}_{dataset}_pretrain_loss.png")
    clustering_tsne_figure_path = os.path.join(
        figure_dir, f"{method}_{dataset}_clustering_tsne.png")
    clustering_umap_figure_path = os.path.join(
        figure_dir, f"{method}_{dataset}_clustering_umap.png")
    clustering_loss_figure_path = os.path.join(
        figure_dir, f"{method}_{dataset}_clustering_loss.png")
    clustering_score_figure_path = os.path.join(
        figure_dir, f"{method}_{dataset}_clustering_score.png")
    if rst_metrics is not None:
        losses_list = []
        loss_names = []
        if len(rst_metrics.Loss) > 1:
            for loss_name in rst_metrics.Loss:
                loss_names.append(loss_name)
                losses_list.append(rst_metrics.Loss[loss_name].val_list)
            gen_loss_chart(losses_list, loss_names,
                           clustering_loss_figure_path)
            figure_paths.append(clustering_loss_figure_path)
        if len(rst_metrics.PretrainLoss) >= 1:
            losses_list = []
            loss_names = []
            for loss_name in rst_metrics.PretrainLoss:
                loss_names.append(loss_name)
                losses_list.append(
                    rst_metrics.PretrainLoss[loss_name].val_list)
            gen_pretrain_loss_chart(
                losses_list, loss_names, pretrain_loss_figure_path)
            figure_paths.append(pretrain_loss_figure_path)
        if true_labels is not None:
            score_dict = {
                'ACC': rst_metrics.ACC.val_list,
                'NMI': rst_metrics.NMI.val_list,
                'ARI': rst_metrics.ARI.val_list,
                'SC': rst_metrics.SC.val_list,
            }
            gen_clustering_chart_metrics_score(
                rst_metrics.Loss['total_loss'].val_list, score_dict, clustering_score_figure_path)
            figure_paths.append(clustering_score_figure_path)

    if features is not None:
        if type(features) == torch.Tensor:
            features = features.cpu().detach().numpy()
        gen_tsne(features, true_labels, pred_labels,
                 clustering_tsne_figure_path)
        gen_umap(features, true_labels, pred_labels,
                 clustering_umap_figure_path)
        figure_paths.append(clustering_tsne_figure_path)
        figure_paths.append(clustering_umap_figure_path)

    if pretrain_features is not None:
        if type(pretrain_features) == torch.Tensor:
            pretrain_features = pretrain_features.cpu().detach().numpy()
        gen_tsne(pretrain_features, true_labels,
                 None, pretrain_tsne_figure_path)
        gen_umap(pretrain_features, true_labels,
                 None, pretrain_umap_figure_path)
        figure_paths.append(pretrain_tsne_figure_path)
        figure_paths.append(pretrain_umap_figure_path)

    return figure_paths


def gen_tsne(features: np.ndarray, true_labels, pred_labels, path, classIdx2label: callable = None):
    # Generate t-SNE visualization of features
    tsne = TSNE(n_components=2, random_state=2023)
    X_tsne = tsne.fit_transform(features)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
    if true_labels is not None:
        if classIdx2label is not None:
            true_labels = classIdx2label(true_labels)
        df_tsne['label1'] = [chr(l + ord('a')) for l in true_labels]
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
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df_tsne, hue='label1', x='Dim1', y='Dim2')
        plt.title('Ground Truth Label TSNE')
        plt.legend(loc='best')
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=df_tsne, hue='label2', x='Dim1', y='Dim2')
        plt.title('Predicted Label TSNE')
        plt.legend(loc='best')
    plt.savefig(path)


def gen_umap(features: np.ndarray, true_labels, pred_labels, path, classIdx2label: callable = None):
    # Generate UMAP visualization of features
    umap = UMAP(n_components=2, random_state=2023)
    X_umap = umap.fit_transform(features)
    df_umap = pd.DataFrame(X_umap, columns=['Dim1', 'Dim2'])
    if true_labels is not None:
        if classIdx2label is not None:
            true_labels = classIdx2label(true_labels)
        df_umap['label1'] = [chr(l + ord('a')) for l in true_labels]
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
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df_umap, hue='label1', x='Dim1', y='Dim2')
        plt.title('Ground Truth Label UMAP')
        plt.legend(loc='best')
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=df_umap, hue='label2', x='Dim1', y='Dim2')
        plt.title('Predicted Label UMAP')
        plt.legend(loc='best')
    plt.savefig(path)


def gen_pretrain_loss_chart(losses_list, loss_names, path, figsize=(10, 6)):
    # Generate pretrain loss chart
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    epochs = list(range(1, len(losses_list[0]) + 1))
    plt.figure(figsize=figsize)
    for i in range(len(losses_list)):
        plt.plot(epochs, losses_list[i], color=colors[i],
                 linewidth=2, label=loss_names[i])
    plt.xlim(1, len(losses_list[0]) + 1)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Pretrain Loss Chart", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.savefig(path)


def gen_loss_chart(losses_list, loss_names, path, figsize=(10, 6)):
    # Generate loss chart
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    epochs = list(range(1, len(losses_list[0]) + 1))
    plt.figure(figsize=figsize)
    for i in range(len(losses_list)):
        plt.plot(epochs, losses_list[i], color=colors[i],
                 linewidth=2, label=loss_names[i])
    plt.xlim(1, len(losses_list[0]) + 1)
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

    plt.xlim(1, len(loss_list) + 1)
    # Plotting for ax1
    loss_line, = ax1.plot(epochs, loss_list, color='steelblue',
                          linewidth=1, label='Total Loss')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)

    # Creating a twin axis for other metrics
    ax2 = ax1.twinx()
    lines = [loss_line]  # Storing the line objects for the legend

    # Plotting for ax2
    if len(score_dict['ACC']) != 0:
        acc_line, = ax2.plot(
            epochs, score_dict['ACC'], color='red', linewidth=1, label='ACC')
        nmi_line, = ax2.plot(
            epochs, score_dict['NMI'], color='orange', linewidth=1, label='NMI')
        ari_line, = ax2.plot(
            epochs, score_dict['ARI'], color='green', linewidth=1, label='ARI')
        lines += [acc_line, nmi_line, ari_line]
    if len(score_dict['SC']) != 0:
        sc_line, = ax2.plot(
            epochs, score_dict['SC'], color='brown', linewidth=1, label='SC')
        lines += [sc_line]

    ax2.set_ylabel("Evaluation Metric", color='black', fontsize=12)

    # Combine legends from both axes
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='best')

    plt.title("Clustering Loss and Evaluation Metrics Chart", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(path)
