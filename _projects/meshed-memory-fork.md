---
title: "Fork of the Meshed Memory Transformer for Image Captioning"
excerpt: "A fork of the Meshed-Memory Transformer for Image Captioning to provide some bug fixes, optimisations, and other tweaks."
collection: projects
github: https://github.com/Delphboy/meshed-memory-transformer/
---

## Overview of the Meshed Memory Transformer
The Meshed Memory Transformer ($$\mathcal{M}2$$) proposed by Cornia *et al.* makes use of the increasingly popular Transformer architecture. Unlike other papers which make use of some predefined structure on extracted image features (spatial graph, semantic graph, etc),  uses stacks of self-attention layers across the set of all the image regions. The standard key and values from the Transformer are edited to include the concatenation of learnable persistent memory vectors. These allow the architecture to encode a-prior knowledge such as "eggs" and "toast" make up the concept "breakfast". When decoding the output of the encoder, a stack of self-attention layers is also used. Each decoder layer is connected via a gated cross attention mechanism to each of the encoder layers, giving way to the "meshed" concept of the paper. The output of the decoder block is used to generate the final output caption.

## Bug Fixes

There is a [bug in the beam search](https://github.com/aimagelab/meshed-memory-transformer/issues/82#issuecomment-1192331647) that has been fixed

## HPC Support

Added HPC support for the QMUL cluster. See the `hpc/` directory for a set of scripts that may be of use.

## Adding the SPICE Evaluation Metric

See [this post for a guide](/words/2024-04-03-adding-spice-to-meshed-memory) on adding the SPICE evaluation metric.