---
title: "FYP Resources"
excerpt: "Some resources to help you with your projects"
collection: fyp
permalink: /fyp/resources
---

# Getting Started with Deep Learning
- **[Hello world, PyTorch style](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)**: The PyTorch team's introduction to deep learning introduces you to all the basic concepts and guides you through training your own neural network
- **[TinyGrad MNIST](https://docs.tinygrad.org/mnist/)** - A guide to solving MNIST (the hello world of deep learning) using TinyGrad
- **[Getting started in JAX](https://docs.jax.dev/en/latest/beginner_guide.html#beginner-guide)** - The JAX team's guide to getting started with JAX.
- **[Getting started with PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html):** For those of you wanting to develop GNNs, start here! They also have some [lecture videos](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html) you may find useful.

# Books

- **[Alice's Adventure in Differentiable Wonderland](https://arxiv.org/pdf/2404.17625):** An introductory text book to differentiable programming (using frameworks like PyTorch or JAX to create neural networks).
- **[The Art of Science and Engineering](https://press.stripe.com/the-art-of-doing-science-and-engineering):** Turing award winner Richard Hamming's Art of Science and Engineering is a cult classic book that covers a wide array of topics, from mathematics to 'style'. Don't be put off by some of the very technical chapters, they can be dense but rewarding. During his time at the US Naval college he also delivered a [lecture series](https://www.youtube.com/watch?v=AD4b-52jtos&list=PL2FF649D0C4407B30) based on his book chapters.

# Websites

- **[Deep learning engineering tricks](/words/2024-05-31-deep-learning-engineering-tricks):** A blog post a wrote a while ago discussing different tricks you can use within your deep learning model to improve its performance.
- **[Scaling Your Model in JAX](https://jax-ml.github.io/scaling-book/)** An online book created by Google DeepMind that provides a rich resource in how to scale your deep learning models.
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/):** A very famous post explaining how the Transformer works under the hood. Recommended reading for anyone starting out in deep learning and wanting to use Transformers.
- **[Transformers are Graph Neural Networks](https://thegradient.pub/transformers-are-graph-neural-networks/):** A very good post explaining an alternative way of viewing how the Transformer works. 


# Lectures

- **Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html) Series:** Andrej Karpathy completed his PhD in image captioning at Stanford before going on to be a founding member of OpenAI. He then moved on to run Tesla's self driving team, before returning to OpenAI. In this lecture series he introduces the foundations of deep learning, starting from backpropogation, and goes all the way to the Transformer. 
- **[3Blue1Brown's Introduction to Deep Learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi):** An excellent animated lecture series introducing the key theory behind deep learning.
- **Richard Hamming's [Learning to Learn](https://www.youtube.com/watch?v=AD4b-52jtos&list=PL2FF649D0C4407B30) Series:** Based on his book [The Art of Science and Engineering](https://press.stripe.com/the-art-of-doing-science-and-engineering), he recaps his career discussing what worked (and what didn't). In later lectures he covers more abstract concepts such as "taste". His lecture titled [You and Your Research](https://www.youtube.com/watch?v=a1zDuOPkMSw&list=PL2FF649D0C4407B30&index=31) is considered a 'must watch' in many circles of computer science. 


# Papers

- **[Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf):** This famous 2017 paper from Google introduces the Transformer architecture, the breakthrough that has lead to the modern AI revolution. If you are working in deep learning, it is essential to understand this paper.
- **[Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf):** This paper introduces the concept of "message passing" for graph neural networks. It presents a framework for "messages", "update functions", and "readout functions" that has been widely adopted in the GNN literature. 
- **[Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929/1000):** This paper introduces the 'Vision Transformer' or 'ViT' and shows how a Transformer can be used to replace traditional computer vision architectures such as CNNs. PyTorch provides pretrained [ViT models](https://docs.pytorch.org/vision/main/models/vision_transformer.html).
- **[Vision GNN (ViG)](https://proceedings.neurips.cc/paper_files/paper/2022/file/3743e69c8e47eb2e6d3afaea80e439fb-Paper-Conference.pdf):**  This paper introduces Vision GNN (ViG) which shows how a GNN can be used instead of a Transformer to provide a vision backbone. It works by constructing a kNN graph over patch features. Section 3.1 contains an interesting discussion on how they minimised the oversmoothing problem associated with GNNs


