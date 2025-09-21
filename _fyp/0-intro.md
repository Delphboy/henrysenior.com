---
title: "Introduction to Geometric Deep Learning FYP Group"
excerpt: "What is (geometric) deep learning anyway? What project should I do?"
collection: fyp
permalink: /fyp/intro
---

# What is "Geometric Deep Learning"?
Deep learning is the practice of learning an approximation of a real world "function" based on existing observations (dataset). Let's say we want to predict whether based on some features whether or not a river is polluted (binary classification). In the real world, there is some magical function, \\(f\\) that exists that looks like \\[y = f(X)\\]It will take some set of features \\(X\\) (i.e. distance from sea, proximity to factories, recent rainfall levels, number of fish) and determine whether or not the river is polluted \\(1=\text{yes}\\) and \\(0=\text{no}\\).

Deep learning is about creating an approximation of this magical function \\(f\\) that closely approximates it. So, we will make our deep learning function \\(g\\) and give it the same features \\(X\\) and get a prediction, \\[\hat y = g(X)\\] which we will then compare against the real world example (ground truth) \\(y\\). If our function \\(g\\) does not match what the real world shows for the same features, we change some of the weights inside \\(g\\) (this will be done for you through a deep learning framework such as [PyTorch](https://pytorch.org/), [JAX](https://docs.jax.dev/en/latest/index.html), or [tinygrad](https://github.com/tinygrad/tinygrad)). This process continues for 100s, 1000s, or 1,000,000s of examples (depending on the size of your dataset) until the function \\(g\\) closely matches the real world \\(f\\).

So, what is **geometric** deep learning? It is a special type of deep learning that is interested in the geometry and structure of the data. Text can be thought of as a line of characters/words, and images can be thought of as a grid. But what about chemical compounds? What about a network of financial transactions? What about your followers and who you follow on instagram? All of these data can be expressed as graphs (which you will be familiar with from your 2nd year DSA module!). Geometric deep learning is about applying deep learning techniques to structures like graphs. It makes use of [Transformers](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)), [Graph Neural Networks](https://en.wikipedia.org/wiki/Graph_neural_network), and [Graph Transformers](https://medium.com/@reutdayan1/graph-transformer-2ede65db4658) to name a few techniques, and is about creating neural networks that focus on the structure of underlying data in order to make predictions. 

# What Kind of Project's Can I Do?
No matter what kind of project you do in deep learning, you will need a dataset. It's recommended that you find one that is already configured for deep learning otherwise you may have to do a lot of data cleaning before you can start assembling your network. You can find datasets in one of two ways: firstly, you can browse [Kaggle](https://www.kaggle.com/datasets). This will show you a lot of smaller niche datasets. Alternatively, you can find a research paper on a topic you find interesting by looking at top deep learning conferences such as [CVPR](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers), [ICCV](https://iccv.thecvf.com/Conferences/2025/AcceptedPapers), [ECCV](https://www.ecva.net/papers.php), [WACV](https://openaccess.thecvf.com/WACV2024), [NeurIPS](https://dblp.org/db/conf/nips/neurips2024.html ), [AAAI](https://aaai.org/proceeding/aaai-39-2025/), [ICML](https://papercopilot.com/paper-list/icml-paper-list/icml-2024-paper-list/), or [ICLR](https://papercopilot.com/paper-list/iclr-paper-list/iclr-2024-paper-list/) or browsing Google Scholar.

See the [projects](project-ideas) page for a list of project ideas to gain inspiration. 

## Research Track
This is the hardest, but most rewarding project type and is ideal for those who want a challenge or are considering further studies in this area. You will work with your supervisors to propose an architecture to solve a particular problem using (geometric) deep learning, playing around with model configurations and running extensive experiments. Who knows, if you do really well, you might even be able to publish your work in a conference or journal!

## ML Engineer Track
This project is about using off the shelf architectures and applying them to a new problem. It still requires all the model training and evaluation, but less of the architectural design of the research track. It can also include taking pretrained models and fine tuning them.

## AI Engineer Track
This involves using existing APIs for large models such as those released by OpenAI, Anthropic, or Google (to name a few). By gluing these together, you will be able to create something that works effectively. Projects of this nature will have to be highly polished in other aspects as calling an API is not sufficiently complex enough for a final year project as any capable first or second year is able to do this. 


# Resources

See the [resources](resources) page I've created

# FAQs
**Can You Give Me a Project Idea?** - Yes, but you might not find it that interesting. No matter what your project, we expect you to spend a lot of time on it, so you may prefer to come up with your own idea that you find interesting. 

**But I'm not good at maths!** - Neither am I, you'll be fine. Do a project you will find interesting, we'll help with any maths you might need!
