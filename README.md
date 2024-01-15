# Clustering Methods Collection & Reproduciton

__Notice: The repository is far from completed, all codes I provided are just for share and will often be changed. And I do not promise any performance and accurate reproduction.__

## Introduction
This project is the collection of the codes reproduced by me and the paper sites about the clustering method in the past years. It also contains many methods which were publicly published and have shared their codes. 

In the recent years, as the representation of unsupervised tasks, the clustering task received great attention from the researchers. Many related works came up and achieved significant success. For most works, the author gave the code for others to use, however, some of them are not fully complete, besides, the running environments and code frameworks are very different or out of fashion. So I decide to collect the papers as well as the codes, and bring them togethor. 

## Install and Usage
1. If you want to use the code provided by this repository, the first thing you need to do is to select a proper position and clone this repository from github. The command you need is:

   ```shell
   git clone git@github.com:Mr-SGXXX/Clustering.git
   cd Clustering
   ```

2. After downloading the repository, you need to construct a proper python environment. I advise you to use the conda, which can easily build a nice environment without influencing your other project setting. You can download Anaconda [here](https://anaconda.org/anaconda/conda), but I suggest you to install miniconda following this [site](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).
   To create a new conda environment, you can use the following command:
   ```shell
   conda create -n clustering python=3.8
   conda activate clustering
   ```

3. If you want to run the codes, you need to install the packages this repository used. And to install them, you can run the following command:
   ```shell
   pip install -r requirements.txt
   ```

4. After preparing the running environment, you can choose which dataset for clustering and which method you want to use in the [config files](./cfg/example.cfg). For the chosen dataset or method, you can also change the hyper-parameters setting in the config file.
   
5. After the experiment, there will be a log file containing the collected message during the experiment as well as a set of figures generated based the features and scores of the experiment. If you stop too many experiments, you can use the following bash script to remove the useless log.
   ```shell
   bash ./clean_log.sh /path/to/log /path/to/figures 
   ```
   Or when you use the default log and figure path, you can use:
   ```shell
   bash ./clean_log.sh
   ```
   **Notice: Do Not Use The Script When Running Any Experiment.**
   
6.  If you want to add a new method or dataset based on this repository, you can firstly lookup the '\_\_init\_\_.py' file for method package (divided into [classical](./methods/classical/__init__.py) methods and [deep](./methods/deep/__init__.py) methods) or [dataset package](./datasetProcesser/__init__.py). Then you can design your method and benefit from the pipeline.


## Methods List
### Classical Methods
Those methods not using deep learning will be included in the this part.

- [ ] [SSC-OMP (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/You_Scalable_Sparse_Subspace_CVPR_2016_paper.html) | [Reference Code]()
- [ ] [EDSC (WACV 2014)](https://ieeexplore.ieee.org/abstract/document/6836065) | [Reference Code]() 
- [ ] [KSSC (ICIP 2014)](https://ieeexplore.ieee.org/abstract/document/7025576) | [Reference Code]()
- [ ] [LRSC (Pattern Recognition Letters 2014)](https://www.sciencedirect.com/science/article/pii/S0167865513003012) | [Reference Code]() 
- [ ] [LRR (ICML 2010)](https://zhouchenlin.github.io/Publications/2010-ICML-LRR.pdf) | [Reference Code]() 
- [ ] [SSC (CVPR 2009)](http://vision.jhu.edu/assets/SSC-CVPR09-Ehsan.pdf) | [Reference Code]() 
- [ ] [DBSCAN (KDD 1996)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) | [Reference Code]() 
- [x] KMeans 
- [x] Spectral Clustering 


### Deep Methods
Those methods using deep learning will be included in this part.
Notice that those muti-view clustering methods and GNN-based clustering methods are not includes here.

- [ ] [DMICC (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26032) | [Reference Code](https://github.com/Li-Hyn/DMICC) 
- [ ] [DivClust (CVPR 2023)](https://arxiv.org/pdf/2304.01042.pdf) | [Reference Code](https://github.com/ManiadisG/DivClust) 
- [ ] [SPICE (TIP 2022)](https://arxiv.org/pdf/2103.09382v1.pdf) | [Reference Code](https://github.com/niuchuangnn/SPICE) 
- [ ] [ProPos (TPAMI 2022)](https://arxiv.org/pdf/2111.11821.pdf) | [Reference Code](https://github.com/Hzzone/ProPos) 
- [ ] [DeepDPM (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.pdf) | [Reference Code](https://github.com/BGU-CS-VIL/DeepDPM) 
- [x] [EDESC (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.pdf) | [Reference Code](https://github.com/JinyuCai95/EDESC-pytorch) | [My Implementation](./methods/deep/EDESC.py)

*In the code provided by the authors, they gave a pretrained weight for Reuters10K, with it, we can gain a nice result sometimes not lower than the article for Reuters10K dataset, but pretraining from start following the code setting in the article instead of using the pretrain weight, the score is hardly as good as what it should be, but similar to this repositary. Besides, the result is not stable.*
- [ ] [VaDeSC (ICLR 2022)](https://openreview.net/pdf?id=RQ428ZptQfU) | [Reference Code](https://github.com/i6092467/vadesc) 
- [ ] [C3-GAN (ICLR 2022)](https://openreview.net/pdf?id=XWODe7ZLn8f) | [Reference Code](https://github.com/naver-ai/c3-gan) 
- [ ] [HC-MGAN (AAAI 2022)](https://arxiv.org/pdf/2112.14772.pdf) | [Reference Code](https://github.com/dmdmello/HC-MGAN) 
- [ ] [MFCVAE (NIPS 2021)](https://arxiv.org/pdf/2106.05241.pdf) | [Reference Code](https://github.com/FabianFalck/mfcvae) 
- [ ] [CLD (CVPR 2021)](http://people.eecs.berkeley.edu/~xdwang/papers/CLD.pdf) | [Reference Code](https://github.com/frank-xwang/CLD-UnsupervisedLearning)
- [ ] [NNM (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Dang_Nearest_Neighbor_Matching_for_Deep_Clustering_CVPR_2021_paper.html) | [Reference Code](https://github.com/ZhiyuanDang/NNM) 
- [ ] [DLRRPD (CVPR 2021)](https://github.com/fuzhiqiang1230/DLRRPD/blob/main/8382_Double_low_rank_representation_with_projection_distance_penalty_for_clustering.pdf) | [Reference Code](https://github.com/fuzhiqiang1230/DLRRPD) 
- [ ] [RUC (CVPR 2021)](https://github.com/fuzhiqiang1230/DLRRPD) | [Reference Code](https://github.com/deu30303/RUC) 
- [ ] [SENet (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Learning_a_Self-Expressive_Network_for_Subspace_Clustering_CVPR_2021_paper.html) | [Reference Code](https://github.com/zhangsz1998/self-expressive-network) 
- [ ] [IDFD (ICLR 2021)](https://openreview.net/pdf?id=e12NDM7wkEY) | [Reference Code](https://openreview.net/pdf?id=e12NDM7wkEY) 
- [ ] [MiCE (ICLR 2021)](https://openreview.net/pdf?id=gV3wdEOGy_V) | [Reference Code](https://github.com/TsungWeiTsai/MiCE) 
- [ ] [CC (AAAI 2021)](https://arxiv.org/pdf/2009.09687.pdf) | [Reference Code](https://github.com/Yunfan-Li/Contrastive-Clustering) 
- [ ] [DFCN (AAAI 2021)](https://arxiv.org/pdf/2012.09600.pdf) | [Reference Code](https://github.com/WxTu/DFCN)
- [ ] [SCCL (NAACL 2021)](https://arxiv.org/pdf/2103.12953.pdf) | [Reference Code](https://github.com/amazon-science/sccl) 
- [ ] [PSSC (TIP 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9440402) | [Reference Code](https://github.com/sckangz/SelfsupervisedSC) 
- [ ] [SCAN (ECCV 2020)](https://arxiv.org/abs/2005.12320) | [Reference Code](https://github.com/wvangansbeke/Unsupervised-Classification) 
- [ ] [IIC (ICCV 2019)](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ji_Invariant_Information_Clustering_for_Unsupervised_Image_Classification_and_Segmentation_ICCV_2019_paper.pdf) | [Reference Code](https://github.com/xu-ji/IIC) 
- [ ] [DEC-DA (ACML 2018)](http://proceedings.mlr.press/v95/guo18b/guo18b.pdf) | [Reference Code](https://github.com/XifengGuo/DEC-DA) 
- [x] [DeepCluster (ECCV 2018)](https://arxiv.org/abs/1807.05520) | [Reference Code](https://github.com/facebookresearch/deepcluster) | [My Implementation](./methods/deep/DeepCluster.py)
- [ ] [SpectralNet (ICLR 2018)](https://openreview.net/pdf?id=HJ_aoCyRZ) | [Reference Code](https://github.com/shaham-lab/SpectralNet) 
- [ ] [DSC-Nets (NIPS 2017)](http://papers.neurips.cc/paper/6608-deep-subspace-clustering-networks.pdf) | [Reference Code](https://github.com/panji1990/Deep-subspace-clustering-networks)
- [ ] [DEPICT (ICCV 2017)](https://arxiv.org/pdf/1704.06327.pdf) | [Reference Code](https://arxiv.org/pdf/1704.06327.pdf) 
- [x] [IDEC (IJCAI 2017)](https://www.researchgate.net/profile/Xifeng-Guo/publication/317095655_Improved_Deep_Embedded_Clustering_with_Local_Structure_Preservation/links/59263224458515e3d4537edc/Improved-Deep-Embedded-Clustering-with-Local-Structure-Preservation.pdf) | [Reference Code](https://github.com/XifengGuo/IDEC) | [My Implementation](./methods/deep/IDEC.py)
  
*In this method, most codes are the same as DEC, except the clustering process. Instead of only using KL loss, the IDEC adds the reconstruct loss in clustering process. Because the IDEC use the same pretrain process as the DEC, in order to save time, the IDEC will directly use the DEC pretrain weight*
- [ ] [VaDE (IJCAI 2017)](https://arxiv.org/pdf/1611.05148.pdf) | [Reference Code](https://github.com/slim1017/VaDE) 
- [ ] [DCN (ICML 2017)](https://arxiv.org/pdf/1610.04794.pdf) | [Reference Code](https://github.com/boyangumn/DCN-New) 
- [x] [DEC (ICML 2016)](https://arxiv.org/pdf/1511.06335.pdf) | [Reference Code](https://github.com/piiswrong/dec/tree/master) | [My Implementaion](./methods/deep/DEC.py)

*In this method, the pretrain process is the most important part, whether the features are learned well by pretraining is directly correspond to whether the result is good. With a reproduced greedy layer-wise pretraining referred to the DEC paper, the pretrained weight is more likely to be good, by which the DEC method is more likely to gain a good score. Though the best score in many experimnets is no lower than the score in the article, the method is still not stable, scores of multiple experiments are very different.*
- [ ] [JULE (CVPR 2016)](https://arxiv.org/pdf/1604.03628.pdf) | [Reference Code](https://github.com/jwyang/JULE.torch)


## Dataset
### Image
- [x] MNIST

- [ ] Fashion MNIST

- [ ] CIFAR-10

- [ ] CIFAR-100

- [ ] STL-10

### Text
- [x] Reuters-10K:
  
*Notice: The Reuters-10K used here is most likely the same as the Reuters-10K used in DEC, which is generated by select random 10000 sample from the [original Reuters](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf) with 685071 samples. Because the original Reuters dataset download url in DEC repository is not available now, the total dataset experiment is not possible for now.*

## Experiment Results
### Results Disclaimer
All the experimental results you can see in this repository are obtained based on the code provided in this repository. Due to factors such as experimental environment and parameter settings, these results may differ slightly or greatly from those in the original paper. I strive to ensure the accuracy of the results, but can't guarantee exact correspondence with the original paper.

The possible difference reasons from my personal view:
- The problem of clustering usually is not stable, the difference of initializing will cause significant difference of results.
- Not all methods were orginally implemented by pytorch, besides different pytorch version may cause difference. This repository may implement the method in a different way.
- Different hardware devices may cause some different results for their slightly different calculating process.
- The results of some methods strictly depends on some weight from a excellent but rare pretrain try, which doesn't occur all the time, causing the scores are easily lower than what authors declaimed.
- Some methods don't offer the hyper-parameter setting they used for all dataset, for these methods we use the default hyper-parameter they offered in their code or paper.
- The public code of some methods can not be run correctly for some bugs or outdated APIs. Though we try to fix these errors, it may cause some diffence of the results.  
- Some methods unfairly used the best epoch recognized by clustering evaluation metrics(ACC, ARI, etc) in the clustering process, which needs ground truth information.(Early stop doesn't mean you can use unfair setting)
- There may be some bugs in this repository which influence the score of some methods. If you find any bug, welcome to raise issues or contact me through email.

The hardware environment accessible to me as follows:
- CPU: AMD EPYC 7302 16-Core Processor
- GPU: NVIDIA GTX 2080ti
- Memory Size: 64GB

### Scores Table
Each method in each dataset will be tried for several times for fair. For deep methods, only the result of the last epoch or the result chosen in a no-need ground truth way will be used. The highest scores as well as mean and std shown as the table with the format "max, mean(std)". The running time of the deep methods contain pretrain time and clustering time.

#### Reuters10K
|       Method        | Test Times |          ACC           |          NMI           |          ARI           |
| :-----------------: | :--------: | :--------------------: | :--------------------: | :--------------------: |
|        EDESC        |     16     | 0.7632, 0.6978(0.0575) | 0.5849, 0.4686(0.0591) | 0.5927, 0.4826(0.0730) |
|         DEC         |     16     | 0.7366, 0.6440(0.0456) | 0.4879, 0.4228(0.0417) | 0.4591, 0.3936(0.0452) |
| Spectral Clustering |     8      | 0.4441, 0.4441(0.0000) | 0.0905, 0.0905(0.0000) | 0.0175, 0.0175(0.0000) |
|       KMeans        |     16     | 0.5622, 0.5301(0.0162) | 0.3549, 0.3243(0.0195) | 0.2655, 0.2211(0.0190) |



## End
In the end, I would like to express my gratitude to all researchers in the field of clustering and the entire AI community for their contributions. Thank you for their willingness to open-source their code. 

In addition, thank to the github for the copilot assistant which greatly improved my efficence.

## Contact
My email is yx_shao@qq.com. If you have any question or advice, please contact me. 