<div align="center">
<h1>Latent Modulated Function for Computational Optimal Continuous Image Representation</h1>
<br>
<a href="https://github.com/HeZongyao">Zongyao He</a><sup><span>1</span></sup>, 
<a href="https://ise.sysu.edu.cn/teacher/teacher02/1384977.htm">Zhi Jin</a><sup><span>1,Corresponding author</span></sup>

<sup>1</sup> SUN YAT-SEN University
<br>
<div>

[![project page](https://img.shields.io/badge/Project-Page-green)](https://github.com/HeZongyao/LMF)
[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)]()
[![poster](https://img.shields.io/badge/Conference-Poster-blueviolet)](https://cvpr.thecvf.com/virtual/2024/poster/31265)
[![license](https://img.shields.io/badge/License-Apache_2.0-blue)](https://opensource.org/licenses/Apache-2.0)

</div>
</div>

## TODO

- [ ] Release the code for Latent Modulated Function (LMF).
- [ ] Release the paper on ArXiv.
- [ ] Provide detailed documentation and usage examples.
- [ ] Include pre-trained models and evaluation results.
- [ ] Add additional features and improvements.

## Introduction

This repository contains the official PyTorch implementation for the CVPR 2024 paper titled "Latent Modulated Function for Computational Optimal Continuous Image Representation" by Zongyao He and Zhi Jin.

<div align="center">
  <img src="assets/efficiency.png" alt="Efficiency comparison" width="50%" />
  <br>

  Efficiency comparisons (320 × 180 input) for ASSR
</div>

<div align="center">
  <img src="assets/framework.png" alt="Framework" />
  <br>

  Framework of our LMF-based continuous image representation
</div>

## Abstract

The recent work Local Implicit Image Function (LIIF) and subsequent Implicit Neural Representation (INR) based works have achieved remarkable success in Arbitrary-Scale Super-Resolution (ASSR) by using MLP to decode Low-Resolution (LR) features. However, these continuous image representations typically implement decoding in High-Resolution (HR) High-Dimensional (HD) space, leading to a quadratic increase in computational cost and seriously hindering the practical applications of ASSR. 

To tackle this problem, we propose a novel Latent Modulated Function (LMF), which decouples the HR-HD decoding process into shared latent decoding in LR-HD space and independent rendering in HR Low-Dimensional (LD) space, thereby realizing the first computational optimal paradigm of continuous image representation. Specifically, LMF utilizes an HD MLP in latent space to generate latent modulations of each LR feature vector. This enables a modulated LD MLP in render space to quickly adapt to any input feature vector and perform rendering at arbitrary resolution. Furthermore, we leverage the positive correlation between modulation intensity and input image complexity to design a Controllable Multi-Scale Rendering (CMSR) algorithm, offering the flexibility to adjust the decoding efficiency based on the rendering precision.

Extensive experiments demonstrate that converting existing INR-based ASSR methods to LMF can reduce the computational cost by up to 99.9%, accelerate inference by up to 57×, and save up to 76% of parameters, while maintaining competitive performance.

## Acknowledgement

This work was supported by [Frontier Vision Lab](https://fvl2020.github.io/fvl.github.com/), SUN YAT-SEN University.

Special acknowledgment goes to the following projects: [LIIF](https://github.com/yinboc/liif), [LTE](https://github.com/jaewon-lee-b/lte), [CiaoSR](https://github.com/caojiezhang/CiaoSR), and [DIIF](https://github.com/HeZongyao/DIIF).

## Citation

If you find this work helpful, please consider citing:

```
@inproceedings{he2024latent,
  title={Latent Modulated Function for Computational Optimal Continuous Image Representation},
  author={He, Zongyao and Jin, Zhi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

Feel free to reach out for any questions or issues related to the code. Thank you for your interest!
