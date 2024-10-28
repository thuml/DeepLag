# DeepLag (NeurIPS 2024)

DeepLag: Discovering Deep Lagrangian Dynamics for Intuitive Fluid Prediction. See [Paper](https://arxiv.org/abs/2402.02425) or [Slides](https://cloud.tsinghua.edu.cn/f/555f0af33db0469ab610/).

This paper proposes using the combination of two perspectives in fluid dynamics, **Eulerian & Lagrangian**, to model the spatio-temporal evolution of fluids, and presents [DeepLag](https://arxiv.org/abs/2402.02425) as a practical Neural Fluid Prediction algorithm, which can bring the following benefits:

- Going beyond learning fluid dynamics at static grids, we propose DeepLag featuring the Eulerian-Lagrangian Recurrent Network, which integrates both Eulerian and Lagrangian frameworks from fluid dynamics within a pure deep learning framework concisely.
- Inspired by Lagrangian mechanics, we present EuLag Block, which can accurately track particle movements and interactively utilize the Eulerian features and dynamic information in fluid prediction, enabling a better dynamic modeling paradigm.
- DeepLag achieves consistent state-of-the art on three representative fluid prediction datasets with superior trade-offs for performance and efficiency, exhibiting favorable practicability.

## Eulerian vs. Lagrangian perspectives

## DeepLag vs. previous methods

## Eulag Block

## Get Started

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