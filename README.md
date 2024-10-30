# DeepLag (NeurIPS 2024)

DeepLag: Discovering Deep Lagrangian Dynamics for Intuitive Fluid Prediction. See [Paper](https://arxiv.org/abs/2402.02425) or [Slides](https://cloud.tsinghua.edu.cn/f/555f0af33db0469ab610/).

This paper proposes using the combination of two perspectives in fluid dynamics, **Eulerian & Lagrangian**, to model the spatio-temporal evolution of fluids, and presents [DeepLag](https://arxiv.org/abs/2402.02425) as a practical Neural Fluid Prediction algorithm, which can bring the following benefits:

- Going beyond learning fluid dynamics at static grids, we propose DeepLag featuring the Eulerian-Lagrangian Recurrent Network, which integrates both Eulerian and Lagrangian frameworks from fluid dynamics within a pure deep learning framework concisely.
- Inspired by Lagrangian mechanics, we present EuLag Block, which can accurately track particle movements and interactively utilize the Eulerian features and dynamic information in fluid prediction, enabling a better dynamic modeling paradigm.
- DeepLag achieves consistent state-of-the art on three representative fluid prediction datasets with superior trade-offs for performance and efficiency, exhibiting favorable practicability.

## Eulerian vs. Lagrangian perspectives

Unlike the Eulerian methods, which observe fluid flow at fixed spatial locations, the Lagrangian approach describes the fluid dynamics through the moving trajectory of individual fluid particles, offering a more natural and neat representation of fluid dynamics with inherent advantages in capturing intricate flow dynamics.

<p align="center">
<img src=".\pic\traj.png" height="150" alt="" align=center />
<br><br>
<b>Figure 1.</b> Comparison between Lagrangian (left) and Eulerian (right) perspectives.
</p>

## DeepLag vs. previous methods

Instead of solely predicting the future based on Eulerian observations, we propose DeepLag to discover hidden Lagrangian dynamics within the fluid by tracking the movements of adaptively sampled key particles. Further, DeepLag presents a new paradigm for fluid prediction, where the Lagrangian movement of the tracked particles is inferred from Eulerian observations, and their accumulated Lagrangian dynamics information is incorporated into global Eulerian evolving features to guide future prediction respectively.

<p align="center">
<img src=".\pic\framework_v4.3.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Three types of neural fluid prediction models (a-c) and overview of DeepLag (d).
</p>

## Eulag Block

We present the EuLag Block, a powerful module that accomplishes Lagrangian tracking and Eulerian predicting at various scales. By leveraging the cross-attention mechanism, the EuLag Block assimilates tracked Lagrangian particle dynamics into the Eulerian field, guiding fluid prediction. It also forecasts the trajectory and dynamics of Lagrangian particles with the aid of Eulerian features.

<p align="center">
<img src=".\pic\eulag_block_v4.3.png" height = "175" alt="" align=center />
<br><br>
<b>Figure 3.</b> Overview of the EuLag Block.
</p>

## Get Started

1. Install Python 3.10 then required packages. For convenience, execute the following command.

```shell
pip install -r requirements.txt
```

2. Download data. 

| Benchmark             | Nickname        | Task                                      |  Download                                                     |
|-----------------------|-----------------|-------------------------------------------| --------------------------------------------------------------|
| Bounded Navier-Stokes | bc              | Predict future dye concentration          | [Link](https://cloud.tsinghua.edu.cn/d/87bb7d8a9778480a90e2/) |
| Ocean Current         | sea             | Predict future marine physical quantities | [Link](https://cloud.tsinghua.edu.cn/d/6e5e65c92a9e425cbbd4/) |
| 3D Smoke              | smoke           | Predict future smoke diffusion            | [Link](https://cloud.tsinghua.edu.cn/d/bf49048a035d4d70b790/) |

Note: you need to use `merge_npy_files` in `utils/split_merge_npy_file.py` to combine the splits of Bounded Navier-Stokes into one `.npy` file before running experiments.

We also provide model checkpoints of DeepLag and all baselines on three benchmarks. If you download them (optional), put them under the folder `./checkpoints/`.

| Model                | Nickname        | Paper                                                    |
|----------------------|-----------------|----------------------------------------------------------|
| U-Net                | unet            | [paper](https://arxiv.org/abs/1505.04597) (MICCAI 2015)  |
| FNO                  | fno             | [paper](https://arxiv.org/abs/2010.08895) (ICLR 2021)    |
| Galerkin Transformer | gktrm           | [paper](https://arxiv.org/abs/2105.14995) (NeurIPS 2021) |
| GNOT                 | gnot            | [paper](https://arxiv.org/abs/2302.14376) (ICML 2023)    |
| LSM                  | lsm             | [paper](https://arxiv.org/abs/2301.12664) (ICML 2023)    |
| Factformer           | factformer      | [paper](https://arxiv.org/abs/2305.17560) (NeurIPS 2023) |
| Vortex               | vortex          | [paper](https://arxiv.org/abs/2301.11494) (ICLR 2023)    |
| **DeepLag (Ours)**   | deeplag         | [paper](https://arxiv.org/abs/2402.02425) (NeurIPS 2024) |

Download Link: [Link](https://cloud.tsinghua.edu.cn/d/4fb89592f8f141c98ca4/). The path format is `{benchmark_nickname}/{model_name}/{time_str}/{model_nickname}_best.pt`

3. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/`. After replacing `--data-path` in the scripts with your local data directory, you can reproduce the experiment results via the following commands:

```shell
bash scripts/bc_deeplag.sh # Train DeepLag on Bounded Navier-Stokes
bash scripts/bc_{baseline_name}.sh # Train baseline on Bounded Navier-Stokes. baseline_name: factformer, fno, gktrm (Galerkin Transformer), gnot, lsm, unet, vortex
bash scripts/sea_deeplag.sh # Train DeepLag on Ocean Current
bash scripts/sea_{baseline_name}.sh # Train baseline on Ocean Current. baseline_name: factformer, fno, gktrm (Galerkin Transformer), gnot, lsm, unet, vortex
bash scripts/smoke_deeplag3d.sh # Train DeepLag on 3D Smoke
bash scripts/smoke_{baseline_name}3d.sh # Train baseline on 3D Smoke. baseline_name: factformer, fno, gktrm (Galerkin Transformer), gnot, lsm, unet
```

To perform standard and long-rollout test with model checkpoints:

```shell
bash test_all.sh
bash test_all_longrollout.sh
```

4. Develop your own model. Here are the instructions:
   - Add the model file under folder `./models/`.
   - Add the model name into `./model_dict.py`.
   - Add a script file under folder `./scripts/` and change the arguments related to the model.

## Results

We extensively experiment on three challenging benchmarks and compare DeepLag with seven baselines. DeepLag achieves the consistent state-of-the-art in both 2D and 3D fluid dynamics (~15% averaged error reduction on **Relative L2**).

| Model                | Bounded Navier-Stokes |         | Ocan Current    |               |  3D Smoke       |
|----------------------|-----------------|---------------|-----------------|---------------|-----------------|
|                      | **10 Frames**   | **30 Frames** | **10 Days**     | **30 Days**   |                 |
| U-Net                | 0.0618          | 0.1038        | 0.0185          | 0.0297        | 0.0508          |
| FNO                  | 0.1041          | 0.1282        | 0.0246          | 0.0420        | 0.0635          |
| Galerkin Transformer | 0.1084          | 0.1369        | 0.0323          | 0.0515        | 0.1066          |
| Vortex               | 0.1999          | NaN           | 0.9548          | NaN           |    -            |
| GNOT                 | 0.1388          | 0.1793        | 0.0206          | 0.0336        | 0.2100          |
| LSM                  | 0.0643          | 0.1020        | 0.0182          | 0.0290        | 0.0527          |
| FactFormer           | 0.0733          | 0.1195        | 0.0183          | 0.0296        | 0.0793          |
| **DeepLag (Ours)**   | **0.0543**      | **0.0993**    | **0.0168**      | **0.0257**    | **0.0378**      |
| **Promotion**        | 13.8%           | 2.7%          | 8.3%            | 12.8%         | 34.4%           |

## Videos of long-term prediction

<p align="center">
<img src=".\pic\bounded-navier-stokes.gif" height = "200" alt="" align=center />
<br><br>
<b>Figure 4.</b> Video of Bounded Navier-Stokes dataset. DeepLag can precisely illustrate the vortex in the center of the figure and give a reasonable motion mode of the Kármán vortex phenomenon formed behind the upper left pillar.
</p>


<p align="center">
<img src=".\pic\ocean-current.gif" height = "200" alt="" align=center />
<br><br>
<b>Figure 5.</b> Video of Ocean Current dataset. DeepLag accurately predicts the location of the high-temperature region to the south area and provides a clear depiction of the Kuroshio pattern.
</p>

## Citation

If you find [DeepLag](https://arxiv.org/abs/2402.02425) or this repo useful, please kindly consider citing our paper.

```
@inproceedings{ma2024deeplag,
  title={DeepLag: Discovering Deep Lagrangian Dynamics for Intuitive Fluid Prediction},
  author={Qilong Ma and Haixu Wu and Lanxiang Xing and Shangchen Miao and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [mql22@mails.tsinghua.edu.cn](mailto:mql22@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/takah29/2d-fluid-simulator