---
title: COCO Semantic Graph Data
collection: words
permalink: /words/2024-05-06coco-semantic-graph-data
excerpt: "A guide to downloading the BUTD and VSUA/SGAE supplementary datasets for COCO."
---

Both the [VSUA paper](http://arxiv.org/pdf/1908.02127) and [SGAE paper](https://arxiv.org/pdf/1812.02378) make use of the same semantic (scene) graph data for the COCO captioning images. This post should explain the files and the data they contain, whilst making it clear how expressively this data can be used. [Data download and extraction instructions can be found here](https://henrysenior.com/words/2024-04-03-coco-supplementary-dataset-download-guide#vsuasgae-data-download)

## What Data Do We Get?
The data is designed to be supplementary to the BUTD data so it builds on top of how that data is structured. In BUTD, each is given a `.npy` file containing a ResNet feature of shape `[X, 2048]`. The `X` refers to the number of objects detected in the image.

Below is an example of the type of `.npy` file provided in the SGAE/VSUA data:
```python
import numpy as np

x = np.load('coco_img_sg/100000.npy',allow_pickle=True, encoding='latin1')
print(x)
########
array({'rela_matrix': 
	array([[  0.,   9., 408.],
       [  0.,  14., 425.],
       [  0.,  26., 425.],
       [  1.,  14., 408.],
       [  1.,  26., 425.],
       [  2.,  22., 410.],
       [  3.,  13., 429.],
       [  3.,  14., 408.],
       [  5.,   7., 410.],
       [  5.,  26., 408.],
       [  7.,  14., 408.],
       [  7.,  26., 408.],
       [ 13.,  14., 463.],
       [ 14.,  26., 408.],
       [ 18.,  20., 408.]]), 
	'obj_attr': 
array([[  0.,  48.,  36., 126., 341., 314., 327.],
       [  1.,  48.,  36.,  49., 341., 314., 327.],
       [  2., 262., 247., 187., 313., 309., 372.],
       [  3., 262., 187., 234., 313., 372., 315.],
       [  4., 227., 218.,  96., 309., 305., 345.],
       [  5., 213., 110.,  40., 313., 350., 336.],
       [  6.,  98., 114.,  38., 313., 332., 315.],
       [  7., 213., 110.,  83., 313., 336., 350.],
       [  8., 227., 218., 255., 305., 325., 309.],
       [  9.,  48.,  36., 126., 314., 341., 327.],
       [ 10., 287., 152.,  73., 313., 380., 401.],
       [ 11., 218., 287., 152., 315., 307., 309.],
       [ 12., 217., 227., 278., 340., 311., 344.],
       [ 13., 262., 187., 276., 313., 309., 372.],
       [ 14., 262.,  48.,  36., 313., 341., 327.],
       [ 15., 213.,  74., 174., 313., 376., 311.],
       [ 16., 213., 174.,  74., 313., 376., 317.],
       [ 17., 213.,  95., 242., 313., 376., 315.],
       [ 18.,  73., 287.,  21., 313., 340., 309.],
       [ 19., 213., 174., 242., 313., 317., 336.],
       [ 20., 287.,  73.,  21., 313., 323., 380.],
       [ 21., 213.,  74.,  95., 313., 376., 317.],
       [ 22., 262.,  98., 114., 313., 323., 380.],
       [ 23., 213., 174.,  74., 313., 376., 317.],
       [ 24., 245., 211.,  34., 313., 323., 377.],
       [ 25., 187., 234., 227., 313., 311., 309.],
       [ 26., 110.,  79.,  28., 313., 327., 307.],
       [ 27., 228., 217., 218., 340., 345., 344.],
       [ 28., 213.,  95., 242., 313., 324., 309.],
       [ 29., 227., 218.,  96., 309., 345., 305.],
       [ 30., 213., 242., 220., 309., 313., 350.]])}, dtype=object)
```

In these files we get two bits of data for each image (in a single file): the relationships between the objects (`'rela_matrix'`) and the attributes attributed to each of objects in the image (`'obj_attr'`). 

## Understanding `rela_matrix`
This portion of the data provides the semantic (scene) graph for the image. It is in the shape of `[Y, 3]` where `Y` is the number of relationships detected in the image. The format is `[index_obj1, index_obj2, rela_id`]. The indexes refer to the indexes of the objects in the original BUTD data object for the same image. (They are named using the COCO image ID so matching the files is relatively straightforward). The `rela_id` however maps to the `coco_pred_sg_rela.npy` file provided by the data. This file contains a two dictionaries that map a label with a numeric ID (and vice versa). The `rela_id` in the `rela_matrix` refers to the ID in this dictionary. In the example above, the 0th object has a relationship with the 14th object that is characterised by the ID `425`. Which, when we look up in `coco_pred_sg_rela.npy` we can see is "under".

## Understanding `obj_attr`
From above, we know that each image has `X` number of objects and `Y` number of relationships between the objects. However, the data also includes additional information about those `X` objects, specifically 6 attributes to further describe the image. 

The `obj_attr` dictionary of the VSUA/SGAE image data contains an `[X, 7]` array where `[:,0]` refers to the index of object in the original BUTD data file for the same image (which explains why the 0th element always counts up). The remaining six elements are all attribute IDs, which can be decoded by looking them up in the `coco_pred_sg_rela.npy` dictionary. Going back to the example above, consider the 0th object, it has attributes: `[48.,  36., 126., 341., 314., 327.]` which maps to: `[carpet, floor, grind, pink, brown, beige]`.

## Incorporating this data into deep learning models
Unlike the BUTD data, this data does not come with predetermined deep model features (i.e. ResNet). Although we can use the BUTD features as object nodes, there are no features for the attributes or relationship between objects. These therefore need to be created either with an `nn.Embedding` table or some other way. How you achieve this is up to you!

## Graph Structure
Finally, onto graph structure! They way that the data means that it is very open and accessible for different graph structures. You can use all the data, or just some of it. You can incorporate relationships as edge features or decide to introduce them as a separate node type. There is no right way to do it, and not necessarily any wrong way of doing it either.