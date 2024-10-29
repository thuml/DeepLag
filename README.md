# DeepLag (NeurIPS 2024)

DeepLag: Discovering Deep Lagrangian Dynamics for Intuitive Fluid Prediction. See [Paper](https://arxiv.org/abs/2402.02425) or [Slides](https://cloud.tsinghua.edu.cn/f/555f0af33db0469ab610/).

This paper proposes using the combination of two perspectives in fluid dynamics, **Eulerian & Lagrangian**, to model the spatio-temporal evolution of fluids, and presents [DeepLag](https://arxiv.org/abs/2402.02425) as a practical Neural Fluid Prediction algorithm, which can bring the following benefits:

- Going beyond learning fluid dynamics at static grids, we propose DeepLag featuring the Eulerian-Lagrangian Recurrent Network, which integrates both Eulerian and Lagrangian frameworks from fluid dynamics within a pure deep learning framework concisely.
- Inspired by Lagrangian mechanics, we present EuLag Block, which can accurately track particle movements and interactively utilize the Eulerian features and dynamic information in fluid prediction, enabling a better dynamic modeling paradigm.
- DeepLag achieves consistent state-of-the art on three representative fluid prediction datasets with superior trade-offs for performance and efficiency, exhibiting favorable practicability.

## Eulerian vs. Lagrangian perspectives

Unlike the Eulerian methods, which observe fluid flow at fixed spatial locations, the Lagrangian approach describes the fluid dynamics through the moving trajectory of individual fluid particles, offering a more natural and neat representation of fluid dynamics with inherent advantages in capturing intricate flow dynamics.

<p align="center">
<img src=".\pic\traj.pdf" height="150" alt="" align=center />
<br><br>
<b>Figure 1.</b> Comparison between Lagrangian (left) and Eulerian (right) perspectives.
</p>

## DeepLag vs. previous methods

Instead of solely predicting the future based on Eulerian observations, we propose DeepLag to discover hidden Lagrangian dynamics within the fluid by tracking the movements of adaptively sampled key particles. Further, DeepLag presents a new paradigm for fluid prediction, where the Lagrangian movement of the tracked particles is inferred from Eulerian observations, and their accumulated Lagrangian dynamics information is incorporated into global Eulerian evolving features to guide future prediction respectively.

<p align="center">
<img src=".\pic\framework_v4.2.pdf" height = "370" alt="" align=center />
<br><br>
<b>Figure 2.</b> Three types of neural fluid prediction models (a-c) and overview of DeepLag (d).
</p>

## Eulag Block

We present the EuLag Block, a powerful module that accomplishes Lagrangian tracking and Eulerian predicting at various scales. By leveraging the cross-attention mechanism, the EuLag Block assimilates tracked Lagrangian particle dynamics into the Eulerian field, guiding fluid prediction. It also forecasts the trajectory and dynamics of Lagrangian particles with the aid of Eulerian features.

<p align="center">
<img src=".\pic\eulag_block_v4.2.pdf" height = "370" alt="" align=center />
<br><br>
<b>Figure 3.</b> Overview of the EuLag Block.
</p>

## Get Started

1. Install Python 3.10 then required packages. For convenience, execute the following command.

```shell
pip install -r requirements.txt
```

2. Download data. If you download model checkpoints (optional), put them under the folder `./checkpoints/`.

3. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```shell
bash scripts/bc_deeplag.sh # Train DeepLag on Bounded Navier-Stokes
bash scripts/bc_{baseline_name}.sh # Train baseline on Bounded Navier-Stokes. baseline_name: factformer, fno, gktrm (Galerkin Transformer), gnot, lsm, unet, vortex
bash scripts/sea_deeplag.sh # Train DeepLag on Ocean Current
bash scripts/sea_{baseline_name}.sh # Train baseline on Ocean Current. baseline_name: factformer, fno, gktrm (Galerkin Transformer), gnot, lsm, unet, vortex
bash scripts/smoke_deeplag3d.sh # Train DeepLag on Ocean Current
bash scripts/smoke_{baseline_name}3d.sh # Train baseline on Ocean Current. baseline_name: factformer, fno, gktrm (Galerkin Transformer), gnot, lsm, unet
```

To perform standard and long-rollout test with model checkpoints:

```shell
bash test_all.sh
bash test_all_longrollout.sh
```

## Results

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

To be continued...