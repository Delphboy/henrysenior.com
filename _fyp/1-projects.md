---
title: "FYP Project Ideas"
excerpt: "It is best if you have your own project idea, but if you need inspiration (or a more 'ready baked' idea) for your project, look no further!"
collection: fyp
permalink: /fyp/project-ideas
---

## Deep Learning Projects

### Image Captioning

There are over 300,000 people in the UK who suffer from (severe) sight impairment (SSI). For these individuals, the online world is increasingly inaccessible due to the amount of visual media that is shared online. Without an alt-text property, screen readers are unable to process images for those with SSI.

Image captioning, the task of using deep learning to produce a humanlike natural language description of an image, aims to solve this problem. Using the [Flickr8K](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset with the appropriate [Karpathy split](https://github.com/Delphboy/karpathy-splits), this project will produce a captioning system that is evaluated on a range of standard captioning benchmarks (BLEU, METEOR, ROGUE, CIDEr, and SPICE).

*Research Track:* Students taking this approach to the project will develop their own model architecture to take in an image and produce the caption. You may start with a simple CNN single image feature encoder coupled with an LSTM, or you may decide to reach for a Transformer encoder-decoder architecture. The choice is yours. This track is ideal for students who want to push themselves and/or are considering further study through a PhD. 

*ML Engineering Track:* Students taking this approach will be expected to build on top of existing models, training "glue" modules where necessary. They may start by either finetuning existing models or creating their own following the CLIP-Cap approach (CLIP image encoder with a thin trainable layer connecting to a GPT-2 language model). This approach is ideal for students who want to learn more about deep learning but are more interested in getting something that works well as a product rather than as a research exercise. 

*AI Engineer Track:* Students taking this approach have little interest in how deep learning actually works and are primarily application programmers. They will use off the shelf APIs (either huggingface or OpenAI et al) to produce the image captions without training a model themselves. In order for this project to be acceptable for a 3rd year dissertation, it must showcase some serious engineering elsewhere in the application. For example, it would be expected to be demonstrated as a marketable product, with a fully functional API that is capable of scaling to handle a large volume of simultaneous requests.

### Comparison of ML Frameworks

PyTorch and JAX are two widely used ML frameworks. However, they are highly complex and have become increasingly specialised towards specific hardware accelerators (CUDA and TPUs respectively). Recently, a newer framework, Tinygrad, has started to gain in popularity. 

The project will involve the creation and training of different neural networks across a range of domains in all three libraries. It will then compare the training and inference performance of these models across different hardware. The project could also be extended to discuss the overall developer experience.


### Remote Sense Change Image Captioning using DINOv3

Remote Sense Imagery (satellite imagery) has become a standard method for monitoring change is areas, tracking issues such as deforestation. Given two remote sensing images of the same location taken at different times, remote sense image change captioning is the task of creating a natural language description of the change that has occurred. 

Meta's recent self-supervised image feature encoder, [DINOv3](https://github.com/facebookresearch/dinov3), has a separate model specifically for remote sense imagery. This project involves using the DINOv3 features to create representations of the remote sense images and using them to inform a language model in order to generate change captions. The project will using the [LEVIR-CC dataset](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) and follow the training regime outlined by [RSCaMa](https://arxiv.org/pdf/2404.18895v1), and be evaluated using the standard benchmark metrics common across the task of image captioning (BLEU, METEOR, ROGUE, CIDEr, and SPICE).

<hr/>

## Non Deep Learning Projects

Whilst this is the *Geometric Deep Learning* FYP group, we know not all of you might want to do a deep learning project. Given that deep learning is our specialism, that is what we are best able to supervise you in. However, we will do our best to supervise projects that fall outside of our domain. Here are some project ideas that don't require deep learning. Remember that a good project typically comes from something you can geek out over. 

### Make a Programming Language

What did you like about Java in the first year? What did you hate about it? Why does it even need a semi-colon?! In this project you will design and build your own programming language. It's up to you whether it's interpreted or compiled, whether it's compatible with the JVM, or whether you decide to do something esoteric like remove all loops. 

For inspriation: [build your own LISP](https://www.buildyourownlisp.com/contents), [original LISP paper](https://www-formal.stanford.edu/jmc/recursive.pdf), [little LISP interpreter](https://maryrosecook.com/blog/post/little-lisp-interpreter), [write a LISP in Python](https://norvig.com/lispy.html), and [write an inerpreter in Go](https://interpreterbook.com/).



