from logging import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from tqdm import tqdm
import numpy as np
import os

from metrics import Metrics
from utils import config

from .backbone.DeepCluster_AlexNet import alexnet
from .backbone.DeepCluster_VGG import vgg16
from .utils.DeepCluster_utils import Kmeans, PIC, cluster_assign
from .base import DeepMethod


class DeepCluster(DeepMethod):
    def __init__(self, dataset, description, logger: Logger, cfg: config):
        super().__init__(dataset, description, logger, cfg)
        backbone = self.cfg.get("DeepCluster", "arch")
        sobel = self.cfg.get("DeepCluster", "sobel")
        if backbone == "alexnet":
            self.model = alexnet(sobel=sobel)
        elif backbone == "vgg16":
            self.model = vgg16(sobel=sobel)
        else:
            raise ValueError(
                f"No available backbone `{backbone}` for DeepCluster")
        self.fd = int(self.model.top_layer.weight.size()[1])
        self.model.top_layer = None
        self.model.features = torch.nn.DataParallel(self.model.features)
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = self.cfg.get("DeepCluster", "batch_size")
        if cfg.get("global", "use_ground_truth_K") and dataset.label is not None:
            n_clusters = dataset.num_classes
        else:
            n_clusters = cfg.get("global", "n_clusters")
            assert type(
                n_clusters) is int, "n_clusters should be of type int"
            assert n_clusters > 0, "n_clusters should be larger than 0"
        clustering = self.cfg.get("DeepCluster", "clustering")
        if clustering == "PIC":
            self.clustering = PIC()
        elif clustering == "Kmeans":
            self.clustering = Kmeans(n_clusters)
        else:
            raise ValueError(
                f"No available clustering `{clustering}` for DeepCluster")

    def forward(self, x):
        pass

    def pretrain(self):
        # the deepcluster is trained directly on the images with kmeans result as label, and doesn't need the pretrain step.
        # the author gives how to donwload the pretrained model
        weight_path = os.path.join(self.cfg.get('global', 'weight_dir'), 'deepcluster_models')
        if not os.path.exists(weight_path):
            self.logger.info("Pretrained model not found, downloading DeepCluster pretrained model...")
            os.system(f"bash ./scripts/download_DeepCluster_model.sh {weight_path}")
        

    def encode_dataset(self):
        self.model.eval()
        train_loader = DataLoader(self.dataset, self.batch_size, shuffle=False)
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
        optimizer = optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.cfg.get("DeepCluster", "lr"),
            momentum=self.cfg.get("DeepCluster", "momentum"),
            weight_decay=10**self.cfg.get("DeepCluster", "weight_decay"),
        )
        criterion = nn.CrossEntropyLoss().to(self.device)
        train_loader = DataLoader(
            self.dataset, self.batch_size, pin_memory=True)
        for epoch in range(self.cfg.get("DeepCluster", "epochs")):
            self.model.train()

            # remove head
            self.model.top_layer = None
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier.children())[:-1])

            # get the features for the whole dataset
            features = self.encode_dataset()

            # cluster the features
            clustering_loss = self.clustering.cluster(
                features.cpu().detach().numpy(), self.logger)

            # assign pseudo-labels
            train_dataset = cluster_assign()
