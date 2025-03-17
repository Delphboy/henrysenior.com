---
title: "COCO Graph Generation"
excerpt: "Precompute various (heterogeneous) scene graphs for the COCO dataset."
collection: projects
github: https://github.com/Delphboy/graph-generation
---

As we show in [our survey](/publications/graph-neural-networks-in-vision-language-image-understanding-a-survey), there are many different ways that an image can be converted to a graph. Whilst there are an increasing number of image captioning projects that make use of graph representations, these graphs are loaded from various [supplementary datasets](/words/2024-05-06-coco-semantic-graph-data) and are assembled as graphs at train/inference time. This codebase aims to provide a single platform for researchers and developers to define a graph structure that matches their downstream model and precompute them. Doing so allows for faster model training and faster development iterations. The extendable pipeline allows for data to be pooled together into either a PyTorch Geometric `Data` or `HeteroData` object that is then saved as a `.pt` file.



