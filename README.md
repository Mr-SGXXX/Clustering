# Clustering
## Install and Usage
1. If you want to use the code provided by this repository, the first thing you need to do is to select a proper position and clone this repository from github. The command you need is:

   ```shell
   git clone git@github.com:Mr-SGXXX/Clustering.git
   ```
2. After downloading the repository, you need to construct a proper python environment.I advise you to use the conda, which can easily build a nice environment without influencing your other project setting. 

## Methods List
### Classical Methods
Those methods not using deep learning will be included in the this part.

- [ ] [SSC-OMP (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/You_Scalable_Sparse_Subspace_CVPR_2016_paper.html) | [Referring Code]()
- [ ] [EDSC (WACV 2014)](https://ieeexplore.ieee.org/abstract/document/6836065) | [Referring Code]() 
- [ ] [KSSC (ICIP 2014)](https://ieeexplore.ieee.org/abstract/document/7025576) | [Referring Code]()
- [ ] [LRSC (Pattern Recognition Letters 2014)](https://www.sciencedirect.com/science/article/pii/S0167865513003012) | [Referring Code]() 
- [ ] [LRR (ICML 2010)](https://zhouchenlin.github.io/Publications/2010-ICML-LRR.pdf) | [Referring Code]() 
- [ ] [SSC (CVPR 2009)](http://vision.jhu.edu/assets/SSC-CVPR09-Ehsan.pdf) | [Referring Code]() 
- [ ] [DBSCAN (KDD 1996)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) | [Referring Code]() 
- [x] KMeans 
- [x] Spectral Clustering 


### Deep Methods
Those methods using deep learning will be included in this part.
Notice that those muti-view clustering methods and GNN-based clustering methods are not includes here.

- [ ] [DMICC (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26032) | [Referring Code](https://github.com/Li-Hyn/DMICC) 
- [ ] [DivClust (CVPR 2023)](https://arxiv.org/pdf/2304.01042.pdf) | [Referring Code](https://github.com/ManiadisG/DivClust) 
- [ ] [SPICE (TIP 2022)](https://arxiv.org/pdf/2103.09382v1.pdf) | [Referring Code](https://github.com/niuchuangnn/SPICE) 
- [ ] [ProPos (TPAMI 2022)](https://arxiv.org/pdf/2111.11821.pdf) | [Referring Code](https://github.com/Hzzone/ProPos) 
- [ ] [DeepDPM (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.pdf) | [Referring Code](https://github.com/BGU-CS-VIL/DeepDPM) 
- [x] [EDESC (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.pdf) | [Referring Code](https://github.com/JinyuCai95/EDESC-pytorch) 

*In the code provided by the authors, they gave a pretrained weight for Reuters10K, with it, we can gain a nice result sometimes not lower than the article for Reuters10K dataset, but pretraining from start following the code setting in the article instead of using the pretrain weight, the score is hardly as good as what it should be, but similar to this repositary. Besides, the result is not stable.*
- [ ] [VaDeSC (ICLR 2022)](https://openreview.net/pdf?id=RQ428ZptQfU) | [Referring Code](https://github.com/i6092467/vadesc) 
- [ ] [C3-GAN (ICLR 2022)](https://openreview.net/pdf?id=XWODe7ZLn8f) | [Referring Code](https://github.com/naver-ai/c3-gan) 
- [ ] [HC-MGAN (AAAI 2022)](https://arxiv.org/pdf/2112.14772.pdf) | [Referring Code](https://github.com/dmdmello/HC-MGAN) 
- [ ] [MFCVAE (NIPS 2021)](https://arxiv.org/pdf/2106.05241.pdf) | [Referring Code](https://github.com/FabianFalck/mfcvae) 
- [ ] [CLD (CVPR 2021)](http://people.eecs.berkeley.edu/~xdwang/papers/CLD.pdf) | [Referring Code](https://github.com/frank-xwang/CLD-UnsupervisedLearning)
- [ ] [NNM (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Dang_Nearest_Neighbor_Matching_for_Deep_Clustering_CVPR_2021_paper.html) | [Referring Code](https://github.com/ZhiyuanDang/NNM) 
- [ ] [DLRRPD (CVPR 2021)](https://github.com/fuzhiqiang1230/DLRRPD/blob/main/8382_Double_low_rank_representation_with_projection_distance_penalty_for_clustering.pdf) | [Referring Code](https://github.com/fuzhiqiang1230/DLRRPD) 
- [ ] [RUC (CVPR 2021)](https://github.com/fuzhiqiang1230/DLRRPD) | [Referring Code](https://github.com/deu30303/RUC) 
- [ ] [SENet (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Learning_a_Self-Expressive_Network_for_Subspace_Clustering_CVPR_2021_paper.html) | [Referring Code](https://github.com/zhangsz1998/self-expressive-network) 
- [ ] [IDFD (ICLR 2021)](https://openreview.net/pdf?id=e12NDM7wkEY) | [Referring Code](https://openreview.net/pdf?id=e12NDM7wkEY) 
- [ ] [MiCE (ICLR 2021)](https://openreview.net/pdf?id=gV3wdEOGy_V) | [Referring Code](https://github.com/TsungWeiTsai/MiCE) 
- [ ] [CC (AAAI 2021)](https://arxiv.org/pdf/2009.09687.pdf) | [Referring Code](https://github.com/Yunfan-Li/Contrastive-Clustering) 
- [ ] [DFCN (AAAI 2021)](https://arxiv.org/pdf/2012.09600.pdf) | [Referring Code](https://github.com/WxTu/DFCN)
- [ ] [SCCL (NAACL 2021)](https://arxiv.org/pdf/2103.12953.pdf) | [Referring Code](https://github.com/amazon-science/sccl) 
- [ ] [PSSC (TIP 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9440402) | [Referring Code](https://github.com/sckangz/SelfsupervisedSC) 
- [ ] [SCAN (ECCV 2020)](https://arxiv.org/abs/2005.12320) | [Referring Code](https://github.com/wvangansbeke/Unsupervised-Classification) 
- [ ] [IIC (ICCV 2019)](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ji_Invariant_Information_Clustering_for_Unsupervised_Image_Classification_and_Segmentation_ICCV_2019_paper.pdf) | [Referring Code](https://github.com/xu-ji/IIC) 
- [ ] [DEC-DA (ACML 2018)](http://proceedings.mlr.press/v95/guo18b/guo18b.pdf) | [Referring Code](https://github.com/XifengGuo/DEC-DA) 
- [ ] [DeepCluster (ECCV 2018)](https://arxiv.org/abs/1807.05520) | [Referring Code](https://github.com/facebookresearch/deepcluster) 
- [ ] [SpectralNet (ICLR 2018)](https://openreview.net/pdf?id=HJ_aoCyRZ) | [Referring Code](https://github.com/shaham-lab/SpectralNet) 
- [ ] [DSC-Nets (NIPS 2017)](http://papers.neurips.cc/paper/6608-deep-subspace-clustering-networks.pdf) | [Referring Code](https://github.com/panji1990/Deep-subspace-clustering-networks)
- [ ] [DEPICT (ICCV 2017)](https://arxiv.org/pdf/1704.06327.pdf) | [Referring Code](https://arxiv.org/pdf/1704.06327.pdf) 
- [ ] [IDEC (IJCAI 2017)](https://www.researchgate.net/profile/Xifeng-Guo/publication/317095655_Improved_Deep_Embedded_Clustering_with_Local_Structure_Preservation/links/59263224458515e3d4537edc/Improved-Deep-Embedded-Clustering-with-Local-Structure-Preservation.pdf) | [Referring Code](https://github.com/XifengGuo/IDEC) 
  
*In this method, most codes are the same as DEC, except the clustering progress. Instead of only using KL loss, the IDEC adds the reconstruct loss in clustering progress. *
- [ ] [VaDE (IJCAI 2017)](https://arxiv.org/pdf/1611.05148.pdf) | [Referring Code](https://github.com/slim1017/VaDE) 
- [ ] [DCN (ICML 2017)](https://arxiv.org/pdf/1610.04794.pdf) | [Referring Code](https://github.com/boyangumn/DCN-New) 
- [x] [DEC (ICML 2016)](https://arxiv.org/pdf/1511.06335.pdf) | [Referring Code](https://github.com/piiswrong/dec/tree/master) 

*In this method, the pretrain progress is the most important part, whether the features are learned well by pretraining will lead to whether the result is good. With a original greedy layer-wise pretraining in DEC paper, the pretrained weight is more likely to be good, by which the DEC method is more likely to gain a good score. Though the best score in many experimnets is no lower than the score in the article, the method is still not stable, scores of multiple experiments are very different.*
- [ ] [JULE (CVPR 2016)](https://arxiv.org/pdf/1604.03628.pdf) | [Referring Code](https://github.com/jwyang/JULE.torch)


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
- Some methods unfairly used the best epoch recognized by clustering evaluation metrics(ACC, ARI, etc) in the clustering progress, which needs ground truth information.(Early stop doesn't mean you can use unfair setting)
- There may be some bugs in this repository which influence the score of some methods. If you find any bug, welcome to raise issues or contact me through email.

The hardware environment accessible to me as follows:
- CPU: AMD EPYC 7302 16-Core Processor
- GPU: NVIDIA GTX 2080ti
- Memory Size: 64GB

### Scores Table
Each method in each dataset will be tried for several times for fair. For deep methods, only the result of the last epoch or the result chosen in a no-need ground truth way will be used. The mean and std shown as the table. The running time of the deep methods contain pretrain time and clustering time.

#### Reuters10K
|       Method        | Test Times |      ACC       |      NMI       |      ARI       | Avg Time |
| :-----------------: | :--------: | :------------: | :------------: | :------------: | :------: |
|        EDESC        |     15     | 0.6765(0.0549) | 0.3806(0.0612) | 0.4271(0.0734) |          |
| Spectral Clustering |            |                |                |                |          |
|       KMeans        |            |                |                |                |          |



## End
In the end, I would like to express my gratitude to all researchers in the field of clustering and the entire AI community for their contributions. Thank you for their willingness to open-source their code. 

In addition, thank to the github for the copilot assistant which greatly improved my efficence.

My email is yx_shao@qq.com. If you have any question or advice, please contact me. 