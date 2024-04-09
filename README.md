<div align="center">
  
# 【IJCAI'2023 🔥】Text-Video Retrieval with Disentangled Conceptualization and Set-to-Set Alignment
  
[![Conference](http://img.shields.io/badge/IJCAI-2023-FFD93D.svg)](https://ijcai-23.org/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2305.12218-FF6B6B.svg)](https://arxiv.org/abs/2305.12218)
</div>

The implementation of IJCAI 2023 paper [Text-Video Retrieval with Disentangled Conceptualization and Set-to-Set Alignment](https://arxiv.org/abs/2305.12218).

## 📌 Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@inproceedings{ijcai2023p0104,
  title     = {Text-Video Retrieval with Disentangled Conceptualization and Set-to-Set Alignment},
  author    = {Jin, Peng and Li, Hao and Cheng, Zesen and Huang, Jinfa and Wang, Zhennan and Yuan, Li and Liu, Chang and Chen, Jie},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {938--946},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/104},
  url       = {https://doi.org/10.24963/ijcai.2023/104},
}

```

<details open><summary>💡 I also have other text-video retrieval projects that may interest you ✨. </summary><p>

> [**Video-Text as Game Players: Hierarchical Banzhaf Interaction for Cross-Modal Representation Learning**](https://arxiv.org/abs/2303.14369)<br>
> Accepted by CVPR 2023 (Highlight) | [[HBI Code]](https://github.com/jpthu17/HBI)<br>
> Peng Jin, Jinfa Huang, Pengfei Xiong, Shangxuan Tian, Chang Liu, Xiangyang Ji, Li Yuan, Jie Chen

> [**DiffusionRet: Generative Text-Video Retrieval with Diffusion Model**](https://arxiv.org/abs/2303.09867)<br>
> Accepted by ICCV 2023 | [[DiffusionRet Code]](https://github.com/jpthu17/DiffusionRet)<br>
> Peng Jin, Hao Li, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, Jie Chen

> [**Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations**](https://arxiv.org/abs/2211.11427)<br>
> Accepted by NeurIPS 2022 | [[EMCL Code]](https://github.com/jpthu17/EMCL)<br>
> Peng Jin, Jinfa Huang, Fenglin Liu, Xian Wu, Shen Ge, Guoli Song, David Clifton, Jie Chen
</p></details>


## 📣 Updates
* **[2023/04/30]**: Release code for reimplementing the experiments in the paper.

## 📕 Overview
Text-video retrieval is a challenging cross-modal task, which aims to align visual entities with natural language descriptions.
Current methods either fail to leverage the local details or are computationally expensive.
What’s worse, they fail to leverage the heterogeneous concepts in data. 
In this paper, we propose the Disentangled Conceptualization and Set-to-set Alignment (DiCoSA) to simulate the conceptualizing and reasoning process of human beings.
For disentangled conceptualization, we divide the coarse feature into multiple latent factors related to semantic concepts.
For set-to-set alignment, where a set of visual concepts correspond to a set of textual concepts, we propose an adaptive pooling method to aggregate semantic concepts to address the partial matching.

<div align="center">
<img src="pictures/pic1.png" width="400px">
</div>

## 📚 Method

<div align="center">
<img src="pictures/pic2.png" width="800px">
</div>

## 🚀 Quick Start
### Datasets
<div align=center>

|Datasets|Google Cloud|Baidu Yun|Peking University Yun|
|:--------:|:--------------:|:-----------:|:-----------:|
| MSR-VTT | [Download](https://drive.google.com/drive/folders/1LYVUCPRxpKMRjCSfB_Gz-ugQa88FqDu_?usp=sharing) | [Download](https://pan.baidu.com/s/1Gdf6ivybZkpua5z1HsCWRA?pwd=enav) | [Download](https://disk.pku.edu.cn/link/AA6A028EE7EF5C48A788118B82D6ABE0C5) |
| MSVD | [Download](https://drive.google.com/drive/folders/18EXLWvCCQMRBd7-n6uznBUHdP4uC6Q15?usp=sharing) | [Download](https://pan.baidu.com/s/1hApFdxgV3TV2TCcnM_yBiA?pwd=kbfi) | [Download](https://disk.pku.edu.cn/link/AA6BD6FC1A490F4D0E9C384EF347F0D07F) |
| ActivityNet | TODO | [Download](https://pan.baidu.com/s/1tI441VGvN3In7pcvss0grg?pwd=2ddy) | [Download](https://disk.pku.edu.cn/link/AAE744E6488E2049BD9412738E14AAA8EA) |
| DiDeMo | TODO | [Download](https://pan.baidu.com/s/1Tsy9nb1hWzeXaZ4xr7qoTg?pwd=c842) | [Download](https://disk.pku.edu.cn/link/AA14E48D1333114022B736291D60350FA5) |

</div>

### Model Zoo
<div align=center>

|Checkpoint|Google Cloud|Baidu Yun|Peking University Yun|
|:--------:|:--------------:|:-----------:|:-----------:|
| MSR-VTT | [Download](https://drive.google.com/file/d/1rMLDHXZw-NgiXnzChkFUIX384crWOCcH/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1CofOx3DLwgNtQcYKYFpf3g?pwd=m9pf) | [Download](https://disk.pku.edu.cn:443/link/16E6BA590227B4580B99AC501C2586B1) |
| ActivityNet | [Download](https://drive.google.com/file/d/1NMnvJs9z0lSX-yFnl-TmbZndq5dO44sw/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1BaAv89II7sIjqKIwBC8QIg?pwd=zu4b) | [Download](https://disk.pku.edu.cn:443/link/9C6D0FC9AEB64928B0F7B9E71B9DD41A) |

</div>


### Setup code environment
```shell
conda create -n DiCoSA python=3.9
conda activate DiCoSA
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

### Download CLIP Model

```shell
cd tvr/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
# wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

### Compress Video
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.


### Test on MSR-VTT
The checkpoint can be downloaded from [pytorch_model.bin.msrvtt](https://disk.pku.edu.cn:443/link/16E6BA590227B4580B99AC501C2586B1).
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=8 \
main_retrieval.py \
--do_eval 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ${OUTPUT_PATH} \
--center 8 \
--temp 3 \
--alpha 0.01 \
--beta 0.005 \
--init_model pytorch_model.bin.msrvtt
```

###  Train on MSR-VTT
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=8 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ${OUTPUT_PATH} \
--center 8 \
--temp 3 \
--alpha 0.01 \
--beta 0.005
```

###  Train on LSMDC
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=8 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 5 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${Anno_PATH} \
--video_path ${DATA_PATH} \
--datatype lsmdc \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ${OUTPUT_PATH} \
--center 8 \
--temp 3 \
--alpha 0.01 \
--beta 0.005
```

###  Train on ActivityNet Captions
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=8 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 10 \
--epochs 10 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${Anno_PATH} \
--video_path ${DATA_PATH} \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir ${OUTPUT_PATH} \
--center 8 \
--temp 3 \
--alpha 0.01 \
--beta 0.005 \
--t2v_beta 50 \
--v2t_beta 50
```

###  Train on DiDeMo
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=8 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${Anno_PATH} \
--video_path ${DATA_PATH} \
--datatype didemo \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir ${OUTPUT_PATH} \
--center 8 \
--temp 3 \
--alpha 0.01 \
--beta 0.005
```

## 🎗️ Acknowledgments
* This code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [DRL](https://github.com/foolwood/DRL), and [EMCL](https://github.com/jpthu17/EMCL).
We sincerely appreciate for their contributions.
