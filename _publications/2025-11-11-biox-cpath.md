---
title: "BioX-CPath: Biologically-driven Explainable Diagnostics for Multistain IHC Computational Pathology"
collection: publications
permalink: /publications/bioxcath
excerpt: "The development of biologically interpretable and explainable models remains a key challenge in computational pathology, particularly for multistain immunohistochemistry (IHC) analysis. We present BioX-CPath, an explainable graph neural network architecture for whole slide image (WSI) classification that leverages both spatial and semantic features across multiple stains. At its core, BioXCPath introduces a novel Stain-Aware Attention Pooling (SAAP) module that generates biologically meaningful, stain-aware patient embeddings. Our approach achieves state-of-the-art performance on both Rheumatoid Arthritis and Sjogren's Disease multistain datasets. Beyond performance metrics, BioX-CPath provides interpretable insights through stain attention scores, entropy measures, and stain interaction scores, that permit measuring model alignment with known pathological mechanisms. This biological grounding, combined with strong classification performance, makes BioX-CPath particularly suitable for clinical applications where interpretability is key. Source code and documentation can be found at: https://github.com/AmayaGS/BioX-CPath."
date: 2025-06-11
venue: 'CVPR'
link: 'https://openaccess.thecvf.com/content/CVPR2025/papers/Gallagher-Syed_BioX-CPath_Biologically-driven_Explainable_Diagnostics_for_Multistain_IHC_Computational_Pathology_CVPR_2025_paper.pdf'
github: https://github.com/AmayaGS/BioX-CPath
citation: 'Amaya Gallagher-Syed, <b>Henry Senior</b>, Omnia Alwazzan, Elena Pontarini, Michele Bombardieri, Costantino Pitzalis, Myles J. Lewis, Michael R. Barnes, Luca Rossi, Gregory Slabaugh; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 10372-10383'
---

## Abstract
The development of biologically interpretable and explainable models remains a key challenge in computational pathology, particularly for multistain immunohistochemistry (IHC) analysis. We present BioX-CPath, an explainable graph neural network architecture for whole slide image (WSI) classification that leverages both spatial and semantic features across multiple stains. At its core, BioXCPath introduces a novel Stain-Aware Attention Pooling (SAAP) module that generates biologically meaningful, stain-aware patient embeddings. Our approach achieves state-of-the-art performance on both Rheumatoid Arthritis and Sjogren's Disease multistain datasets. Beyond performance metrics, BioX-CPath provides interpretable insights through stain attention scores, entropy measures, and stain interaction scores, that permit measuring model alignment with known pathological mechanisms. This biological grounding, combined with strong classification performance, makes BioX-CPath particularly suitable for clinical applications where interpretability is key.

### Key Features
We introduce BioX-CPath, an interpretable graph neural network for multistain pathology that achieves state-of-the-art performance while providing biological insights through a novel stain-aware attention mechanism.

- **Stain-Aware Attention Pooling (SAAP)**: A novel module that generates biologically meaningful, stain-aware patient embeddings
- **State-of-the-art performance**: Validated on Rheumatoid Arthritis and Sjogren's Disease multistain datasets
- **Interpretable insights**: Provides stain attention scores, entropy measures, and stain interaction scores and GNN node heatmaps
- **Biological alignment**: Model outputs can be measured against known pathological mechanisms
- **Clinical applicability**: Combines strong classification performance with the interpretability needed for clinical settings

![](https://github.com/AmayaGS/BioX-CPath/blob/main/figures/pipeline_final.png)


## Citation

```bibtex
@inproceedings{gallagher2025biox,
  title={BioX-CPath: Biologically-driven Explainable Diagnostics for Multistain IHC Computational Pathology},
  author={Gallagher-Syed, Amaya and Senior, Henry and Alwazzan, Omnia and Pontarini, Elena and Bombardieri, Michele and Pitzalis, Costantino and Lewis, Myles J and Barnes, Michael R and Rossi, Luca and Slabaugh, Gregory},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10372--10383},
  year={2025}
}
```


