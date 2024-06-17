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

# This method reproduction refers to the following repository:
# https://github.com/facebookresearch/deepcluster
from logging import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from tqdm import tqdm
import numpy as np
import os

from datasetLoader import ClusteringDataset, reassign_dataset
from metrics import normalized_mutual_info_score as cal_nmi
from utils import config

from .DeepCluster_AlexNet import alexnet
from .DeepCluster_VGG import vgg16
from .DeepCluster_utils import Kmeans, PIC, UnifLabelSampler, arrange_clustering
from ..base import DeepMethod


class DeepCluster(DeepMethod):
    def __init__(self, dataset:ClusteringDataset, description:str, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        self.backbone = self.cfg.get("DeepCluster", "arch")
        sobel = self.cfg.get("DeepCluster", "sobel")
        if self.backbone == "alexnet":
            self.model = alexnet(sobel=sobel)
        elif self.backbone == "vgg16":
            self.model = vgg16(sobel=sobel)
        else:
            raise ValueError(
                f"No available backbone `{self.backbone}` for DeepCluster")
        self.fd = int(self.model.top_layer.weight.size()[1])
        self.resume = self.cfg.get("DeepCluster", "resume")
        self.model.top_layer = None
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.reassign = self.cfg.get("DeepCluster", "reassign")
        self.batch_size = self.cfg.get("DeepCluster", "batch_size")
        clustering = self.cfg.get("DeepCluster", "clustering")
        if clustering == "PIC":
            self.clustering = PIC()
        elif clustering == "Kmeans":
            self.clustering = Kmeans(self.n_clusters)
        else:
            raise ValueError(
                f"No available clustering `{clustering}` for DeepCluster")


    def pretrain(self):
        # the deepcluster is trained directly on the images with kmeans result as label, and doesn't need the pretrain step.
        # the author gives how to donwload the pretrained model
        if self.resume is not None and self.resume == "download":
            weight_path = os.path.join(self.cfg.get('global', 'weight_dir'), 'deepcluster')
            self.resume = os.path.join(weight_path, self.backbone)
            self.resume = os.path.join(weight_path, 'checkpoint.pth.tar')
            if not os.path.exists(self.resume):
                self.logger.info("Pretrained model not found, downloading DeepCluster pretrained model...")
                os.system(f"bash ./scripts/download_DeepCluster_model.sh {weight_path}")
            else:
                self.logger.info("Pretrained model found, skipping downloading...")
        return None
            

    def encode_dataset(self):
        self.model.eval()
        train_loader = DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=self.workers)
        latent_list = []
        with torch.no_grad():
            for data, _, _ in tqdm(train_loader, desc="Encoding dataset", dynamic_ncols=True, leave=False):
                data = data.to(self.device)
                aux = self.model(data)
                latent_list.append(aux)
        latent = torch.cat(latent_list, dim=0)
        self.train()
        return latent

    def train_model(self):
        learning_rate = self.cfg.get("DeepCluster", "learn_rate")
        weight_decay = self.cfg.get("DeepCluster", "weight_decay")
        optimizer = optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.cfg.get("DeepCluster", "lr"),
            momentum=self.cfg.get("DeepCluster", "momentum"),
            weight_decay=10**self.cfg.get("DeepCluster", "weight_decay"),
        )
        criterion = nn.CrossEntropyLoss().to(self.device)
        start_epoch = 0
        if self.resume is not None:
            if os.path.isfile(self.resume):
                self.logger.info("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                start_epoch = checkpoint['epoch']
                # remove top_layer parameters from checkpoint
                for key in checkpoint['state_dict']:
                    if 'top_layer' in key:
                        del checkpoint['state_dict'][key]
                self.model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.resume, checkpoint['epoch']))
            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.resume))

        y_pred_last = None
        delta_label = 0
        with tqdm(range(start_epoch, self.cfg.get("DeepCluster", "epochs")), desc="Training DeepCluster Epoch", dynamic_ncols=True, leave=False) as epochloader:
            for epoch in epochloader:
                self.model.train()

                # remove head
                self.model.top_layer = None
                self.model.classifier = nn.Sequential(
                    *list(self.model.classifier.children())[:-1])

                # get the features for the whole dataset
                features = self.encode_dataset()

                # cluster the features
                clustering_loss = self.clustering.cluster(features.cpu().detach().numpy(), self.logger)
                y_pred = arrange_clustering(self.clustering.images_lists)

                # evaluate clustering performance
                if self.cfg.get("global", "record_sc"):
                    _, (acc, nmi, ari, _, _) = self.metrics.update(y_pred, features, y_true=self.dataset.label)
                else:
                    _, (acc, nmi, ari, _, _) = self.metrics.update(y_pred, y_true=self.dataset.label)

                # assign pseudo-labels
                train_dataset = reassign_dataset(self.dataset, self.clustering.images_lists)

                # uniformly sample per target
                sampler = UnifLabelSampler(int(self.reassign * len(train_dataset)),
                                    self.clustering.images_lists)
                
                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.workers, pin_memory=True)

                # set last fully connected layer
                mlp = list(self.model.classifier.children())
                mlp.append(nn.ReLU(inplace=True).to(self.device))
                self.model.classifier = nn.Sequential(*mlp)
                self.model.top_layer = nn.Linear(self.fd, len(self.clustering.images_lists))
                self.model.top_layer.weight.data.normal_(0, 0.01)
                self.model.top_layer.bias.data.zero_()
                self.model.top_layer.to(self.device)

                # train network with clusters as pseudo-labels
                optimizer_tl = optim.SGD(
                    self.model.top_layer.parameters(),
                    lr=learning_rate,
                    weight_decay=10**weight_decay,
                )
                total_loss = 0
                with tqdm(train_dataloader, desc="Training DeepCluster Batch", dynamic_ncols=True, leave=False) as batchloader:
                    for i, (input_tensor, target_tensor, _) in enumerate(batchloader):
                        input_tensor = input_tensor.to(self.device)
                        target_tensor = target_tensor.to(self.device)
                        target_tensor = torch.squeeze(target_tensor)

                        output = self.model(input_tensor)
                        loss = criterion(output, target_tensor)

                        # compute gradient and do SGD step
                        optimizer.zero_grad()
                        optimizer_tl.zero_grad()
                        loss.backward()
                        total_loss += loss.item()
                        optimizer.step()
                        optimizer_tl.step()
                        
                        batchloader.set_postfix({
                            "Loss": loss.item()
                        })

                        if i % 200 == 0:
                            self.logger.info(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")
                
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                delta_nmi = cal_nmi(y_pred, y_pred_last)
                self.logger.info(f'Epoch {epoch + 1}\tAcc {acc:.4f}\tNMI {nmi:.4f}\tARI {ari:.4f}\tDelta Label {delta_label:.4f}\tDelta NMI {delta_nmi:.4f}\n')
                self.logger.info(f"Clustering Loss: {clustering_loss}\tConvNet Loss: {total_loss / len(train_dataloader):.4f}")
                y_pred_last = y_pred

                epochloader.set_postfix({
                    "ACC": acc,
                    "NMI": nmi,
                    "ARI": ari,
                    "Delta Label": delta_label,
                    "Delta NMI": delta_nmi
                })
                # save running checkpoint
                weight_dir = os.path.exists(os.path.join(self.cfg.get("global", "weight_dir"), "deepcluster_checkpoints"))
                if not weight_dir:
                    os.makedirs(weight_dir)
                torch.save({'epoch': epoch + 1,
                    'arch': self.backbone,
                    'state_dict': self.model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                os.path.join(weight_dir, f'checkpoint_{self.backbone}_{self.description}.pth.tar'))
        clustering_loss = self.clustering.cluster(features.cpu().detach().numpy(), self.logger)
        y_pred = arrange_clustering(self.clustering.images_lists)
        return y_pred, features
                
