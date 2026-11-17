---
title: "Joint Embeddings: From CLIP to SigLIP and BLIP"
excerpt: "Build a small image–text matcher in PyTorch. Then learn how CLIP, SigLIP, BLIP, and BLIP-2 work."
collection: fyp
permalink: /fyp/joint-embeddings
---

How can a model answer **“which of these pictures contains a red fox?”** if the training data had no fixed category “red fox”? One answer is this: put images and text into the same meaningful space, then compare them.

This tutorial builds that idea from the start. You will train a small CLIP-like model on four coloured images and four descriptions. Then you will inspect its similarity matrix. After that, you will see how SigLIP, BLIP, and BLIP-2 change the design. You need basic Python. You do not need knowledge of Transformers. If tensors, losses, and training loops are new to you, read the [PyTorch MNIST tutorial](pytorch-tutorial) first.

> **Slides:** [Download the slides here](/files/fyp/joint-embeddings-slides.pptx).

**When you complete this page, you can:**

- Explain an embedding and a *joint* embedding space.
- Calculate the symmetric contrastive loss of CLIP.
- Build and test a very small dual encoder in PyTorch.
- Explain zero-shot classification correctly.
- Select a dual encoder, a joint image–text matcher, or a caption-generating model.

## 1. The problem: categories are closed, language is open

A standard image classifier has one output score for each category in its training data:

```text
image ──► neural network ──► [cat: 0.1, dog: 0.8, car: 0.1]
```

To add a new category, you must add an output and train the network again. The network also cannot use a description such as “a dog running through snow”. It knows only its fixed outputs.

So ask a different question: **how well does this image agree with this text?**

```text
image ──► image encoder ──► image embedding ─┐
                                             ├─► similarity score
text  ──► text encoder  ──► text embedding  ─┘
```

An **embedding** is a vector of numbers that represents an input. A 2-dimensional embedding can look like `[0.8, -0.2]`. Real models often use hundreds or thousands of dimensions. The model learns the coordinates. A person does not give a coordinate a meaning such as “redness” or “furriness”.

A **joint embedding space** is a shared coordinate system for different types of input. The image encoder and the text encoder are different functions. But you can compare their output vectors directly:

$$
\mathbf{v} = f_{\text{image}}(\text{image}), \qquad
\mathbf{t} = f_{\text{text}}(\text{text}).
$$

“Joint” refers to the **aligned output space**. It does not mean that images and text use the same encoder or the same weights.

The useful property is this:

> An image and a text that match have similar embeddings. An image and a text that do not match have a lower similarity.

The model learns this arrangement from examples. We do not tell the model where “red” or “dog” must go.

## 2. The mathematics: length, dot product, cosine similarity

To calculate the **dot product** of two vectors, multiply the matching coordinates. Then add the results:

$$
\mathbf{v} \cdot \mathbf{t} = \sum_k v_k t_k.
$$

The result is large when the vectors point in a similar direction. But the result also depends on the vector lengths. As a result, CLIP **normalises** each embedding to length 1:

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2}.
$$

For these unit-length vectors, the dot product is the **cosine similarity**:

$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{t}}{\lVert\mathbf{v}\rVert_2\lVert\mathbf{t}\rVert_2}
= \hat{\mathbf{v}}\cdot\hat{\mathbf{t}},
$$

The value is between \\(-1\\) (opposite) and \\(1\\) (same direction). A value of \\(0\\) shows perpendicular vectors. For example, `[1, 0]` and `[0, 1]` have similarity 0. `[1, 0]` and `[-1, 0]` have similarity -1.

Normalisation lets training change the **direction** of a vector. The model cannot increase a score only by making all vectors longer.

## 3. CLIP: contrastive language–image pre-training

[CLIP](https://arxiv.org/abs/2103.00020) (OpenAI, 2021) has two main parts:

1. An **image encoder**, for example a Vision Transformer or a ResNet.
2. A **text encoder**. This is a Transformer that processes a tokenised caption.

Each encoder makes features. The model projects the features to the same embedding width and normalises them. The original model learned from 400 million image–text pairs from the internet. Our small model uses the same core idea with four pairs.

### 3.1 One batch becomes a matching problem

Suppose that a batch has four pairs:

```text
image 0 ↔ “a red square”
image 1 ↔ “a green square”
image 2 ↔ “a blue square”
image 3 ↔ “a yellow square”
```

Compare **each image with each caption**. Do not compare an image only with its partner. Matrix \\(V\\) contains the image embeddings as rows. Matrix \\(T\\) contains the text embeddings as rows. Then \\(VT^\top\\) gives this matrix:

| Similarity | red text | green text | blue text | yellow text |
|---|---:|---:|---:|---:|
| **red image** | **correct** | negative | negative | negative |
| **green image** | negative | **correct** | negative | negative |
| **blue image** | negative | negative | **correct** | negative |
| **yellow image** | negative | negative | negative | **correct** |

The correct pairs are on the **diagonal**. Training makes the diagonal scores higher than the off-diagonal scores. We call this method **contrastive learning**. The model contrasts correct pairs with incorrect alternatives.

The other captions in a batch are **in-batch negatives**. This method gives many training comparisons. You do not need to label each possible pair by hand.

### 3.2 From scores to a loss

For each row, **softmax** changes the scores to a distribution across the captions. Then **cross-entropy loss** increases the penalty when the model gives too little weight to the diagonal entry. You can see this as a classification problem. For row \\(i\\), the correct class is caption \\(i\\).

Make the same calculation for each column. Retrieval must work in two directions:

- **Image → text:** given an image, find its caption.
- **Text → image:** given a caption, find its image.

The similarity logits include a learned scale:

$$
s_{ij} = \exp(a)\,\hat{\mathbf{v}}_i\cdot\hat{\mathbf{t}}_j
       = \frac{\hat{\mathbf{v}}_i\cdot\hat{\mathbf{t}}_j}{\tau}.
$$

Here \\(a\\) is a learned parameter with the name `logit_scale`. \\(\tau\\) is the **temperature**. A larger scale makes the softmax distributions sharper. It does **not** make the cosine similarity a calibrated probability.

For one direction with \\(B\\) pairs, the loss is:

$$
\mathcal{L}_{i\rightarrow t} = -\frac{1}{B}\sum_i
\log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ij})}.
$$

CLIP averages this loss with the transposed direction, \\(\mathcal{L}_{t\rightarrow i}\\). This symmetric form of **InfoNCE** is a contrastive objective.

Here is the complete loss function in PyTorch:

```python
import torch
import torch.nn.functional as F


def clip_loss(image_embeddings, text_embeddings, scale):
    # Rows are examples. Columns are embedding coordinates.
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # [batch, embedding] @ [embedding, batch] -> [batch, batch]
    logits = scale * image_embeddings @ text_embeddings.T
    targets = torch.arange(logits.shape[0], device=logits.device)

    image_to_text = F.cross_entropy(logits, targets)
    text_to_image = F.cross_entropy(logits.T, targets)
    return (image_to_text + text_to_image) / 2
```

Use `cross_entropy` directly on the logits. This function does the log-softmax calculation in a numerically stable way. Do not use a batch with one pair. It has no alternatives for the model to learn from.

## 4. Build a small CLIP-like retriever

Next, train two small neural networks. The networks match four solid-colour images to four captions. The code runs on a CPU and needs only PyTorch.

This is **not a small copy of the data or the ability of CLIP**. It is a **mechanism demonstrator**. The encoders, the normalisation, the similarity matrix, and the two-way loss are real. The visual understanding and the language understanding are not.

Create the file `tiny_clip.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

# --- 1. Make four 2x2 RGB images -------------------------------------------
names = ["red", "green", "blue", "yellow"]
colours = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def solid_image(rgb):
    # [3] -> [3, 1, 1] -> [3, 2, 2]
    return torch.tensor(rgb).reshape(3, 1, 1).repeat(1, 2, 2)


images = torch.stack([solid_image(colours[name]) for name in names])
captions = [f"a photo of a {name} square" for name in names]

# --- 2. Make bag-of-words vectors from the captions -----------------------
vocabulary = sorted({word for caption in captions for word in caption.split()})
word_to_index = {word: i for i, word in enumerate(vocabulary)}


def bag_of_words(caption):
    vector = torch.zeros(len(vocabulary))
    for word in caption.split():
        vector[word_to_index[word]] += 1
    return vector


text_inputs = torch.stack([bag_of_words(caption) for caption in captions])

# --- 3. Define two encoders and a learned inverse temperature --------------
class TinyCLIP(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim=16):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Flatten(),                 # [3, 2, 2] -> [12]
            nn.Linear(12, 32), nn.ReLU(),
            nn.Linear(32, embedding_dim), # projection into the joint space
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(vocabulary_size, 32), nn.ReLU(),
            nn.Linear(32, embedding_dim), # projection into the same space
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, text_inputs):
        image_embeddings = F.normalize(self.image_encoder(images), dim=-1)
        text_embeddings = F.normalize(self.text_encoder(text_inputs), dim=-1)
        return image_embeddings, text_embeddings, self.logit_scale.exp()


def clip_loss(image_embeddings, text_embeddings, scale):
    logits = scale * image_embeddings @ text_embeddings.T
    targets = torch.arange(logits.shape[0])
    return (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    ) / 2


model = TinyCLIP(len(vocabulary))
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 4. Train on all four pairs as one batch -------------------------------
for step in range(1, 301):
    image_embeddings, text_embeddings, scale = model(images, text_inputs)
    loss = clip_loss(image_embeddings, text_embeddings, scale)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if step == 1 or step % 100 == 0:
        print(f"step {step:03d} | loss {loss.item():.3f}")

# --- 5. Retrieve a caption for each image, and the opposite ----------------
model.eval()
with torch.no_grad():
    image_embeddings, text_embeddings, _ = model(images, text_inputs)
    similarity = image_embeddings @ text_embeddings.T
    image_to_text = similarity.argmax(dim=1)
    text_to_image = similarity.argmax(dim=0)

print("\nCosine similarities (rows=images, columns=captions):")
print(similarity.round(decimals=2))

print("\nImage-to-text retrieval:")
for i, name in enumerate(names):
    print(f"{name:6s} image -> {captions[image_to_text[i]]}")

expected = torch.arange(len(captions))
assert torch.equal(image_to_text, expected)
assert torch.equal(text_to_image, expected)
print("\nBoth retrieval directions matched all four pairs.")
```

Run the file with this command:

```bash
python tiny_clip.py
```

Your exact numbers can be different in other PyTorch versions. But the loss must go down. Also, each caption must retrieve its matching colour. The assertions check this result. Do not judge the result only from a loss that goes down.

### 4.1 The function of each part

- **The images are tensors.** Each image has the shape `[3, 2, 2]`. This shape is three colour channels and a 2×2 pixel grid. The batch has the shape `[4, 3, 2, 2]`.
- **The encoders are small neural networks.** `Linear` is a learned matrix multiplication and a bias. `ReLU` calculates `max(0, x)`. This non-linearity lets the stack of layers learn more than one linear transformation.
- **The text is also a tensor.** A bag-of-words vector counts each vocabulary word in a caption. It discards the word order and the grammar. Real CLIP uses token embeddings and a Transformer.
- **The last `Linear` layers are projections.** They map two different hidden representations into the same 16-dimensional space. `F.normalize` changes each output to a unit vector.
- **`@ text_embeddings.T` makes all comparisons.** The result is the 4×4 matrix from section 3.1.
- **`targets = [0, 1, 2, 3]`.** Row 0 must select column 0. Row 1 must select column 1, and so on. The transposed logits change the retrieval direction.
- **Gradients go to both encoders.** A bad score changes the image weights and the text weights. As a result, the outputs become more compatible.

We evaluate on the **training pairs** on purpose. This test shows that your implementation of the mechanism is correct. It does not show generalisation. The model can memorise four templates. The vocabulary comes from the same captions. The model cannot write useful captions for photographs.

> **Note:** Inspect the shapes. Add `print(images.shape, text_inputs.shape)` before training. Add `print(image_embeddings.shape, logits.shape)` in the loss function. If you can explain each dimension, you understand the mechanism better. A larger network does not give this understanding.

## 5. The meaning of “zero-shot” classification

After you make a joint space, classification can become retrieval:

1. Change each possible class to a sentence, for example `a photo of a dog` and `a photo of a cat`.
2. Encode these prompts with the text encoder.
3. Encode the new image with the image encoder.
4. Select the prompt with the highest cosine similarity. Or apply softmax to the scaled similarities to get relative class scores.

**Zero-shot does not mean that the model was never trained.** It means that this classification task needs no more labelled examples and no gradient updates. CLIP still had large-scale pre-training. Its performance depends on the concepts in that data.

This fact also shows why prompts are important. “a photo of a dog” and “a grainy black-and-white photograph of a dog” can give different results. Zero-shot transfer is not always accurate. It is not always fair.

## 6. SigLIP: keep the two encoders, change the loss

[SigLIP](https://arxiv.org/abs/2303.15343) (2023) keeps the dual-encoder design with a shared embedding. Its main change is the **pairwise sigmoid loss**.

CLIP asks one question: “which caption in this row matches?” SigLIP asks a separate question for each cell: “does this image–text pair match?” A match has the label +1. A non-match has the label -1.

For a pair with logit \\(z_{ij}\\) and label \\(y_{ij}\in\{-1,+1\}\\), the loss contribution is:

$$
-\log \sigma(y_{ij}z_{ij}),
$$

Here \\(\sigma\\) is the sigmoid function. For a correct, confident pair, \\(y_{ij}z_{ij}\\) is large and positive. Its loss is near zero.

You can use this loss instead of the CLIP loss:

```python
def siglip_loss(image_embeddings, text_embeddings, scale, bias):
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    logits = scale * image_embeddings @ text_embeddings.T + bias

    # +1 on the diagonal, -1 in all other cells.
    targets = 2 * torch.eye(logits.shape[0], device=logits.device) - 1
    return -F.logsigmoid(targets * logits).mean()
```

In SigLIP, `scale` and `bias` are learned. The important difference is this. The loss of each pair does **not** need a softmax normalisation across the batch. As a result, the objective is easier to distribute. It can also give better results with small batches.

It does **not** mean these things:

- There are no negatives (the off-diagonal pairs are still negatives).
- The contents of the batch are not important (the mini-batches still supply the negatives).
- Each sigmoid score is a calibrated probability of a match.
- Softmax contrastive learning is always worse.

### SigLIP 2: more training, same encoder interface

[SigLIP 2](https://arxiv.org/abs/2502.14786) (2025) keeps the sigmoid image–text objective. It adds these training methods:

- Image-grounded **captioning and localisation objectives** with an auxiliary decoder.
- **Self-distillation and masked-image prediction** for better local or “dense” visual features.
- Multilingual data, online data selection, and variants for different resolutions and native aspect ratios.

The aim is better semantic understanding, localisation, multilingual performance, and dense features. It is not only a different matching loss. Remember this difference. Captioning helped to train the representation. But the standard released SigLIP 2 encoder is not a caption generator.

## 7. BLIP: retrieval, joint matching, and generation

A dual encoder compares two vectors and stops there. It cannot write a new caption. [BLIP](https://arxiv.org/abs/2201.12086) (**Bootstrapping Language–Image Pre-training**, 2022) has a wider design. It supports understanding and it supports generation.

BLIP uses a vision encoder and a text module. The text module has three modes:

| Component | Objective | What it learns |
|---|---|---|
| Separate image encoder and text encoder | **ITC:** image–text contrastive | Align embeddings for fast retrieval |
| Image-grounded text encoder | **ITM:** image–text matching | Inspect a pair and classify match or non-match |
| Image-grounded text decoder | **LM:** language modelling | Generate a caption one token at a time, with the image as condition |

The ITM path uses **cross-attention**. As a result, the text features can attend to the image features. This operation is not a CLIP dot product. It can examine a candidate pair more closely. But you cannot precompute one vector for each item and compare millions of candidates with one cheap matrix operation. So a fast dual-encoder retrieval stage can supply candidates. Then a more expensive matcher can rank them again.

### The “bootstrapping” part is also data work

Captions from the web are noisy. BLIP uses the **CapFilt** method:

1. A **captioner** makes synthetic captions for web images.
2. A **filter** uses image–text matching to reject noisy original captions and noisy synthetic captions.
3. The better mixture of filtered web data and human-labelled data pre-trains a new model.

As a result, BLIP is not only “CLIP with a decoder”. Its model objectives **and its data-bootstrapping method** are important.

## 8. BLIP-2: train a bridge between frozen models

Large vision models and language models are expensive to train together. [BLIP-2](https://arxiv.org/abs/2301.12597) (2023) starts with a pretrained image encoder and a pretrained large language model (LLM). Then it keeps both models **frozen** during pre-training. Their weights get no gradient updates.

Between the two models is a small **Querying Transformer (Q-Former)**:

```text
image ──► frozen image encoder ──► image features ─┐
                                                   ▼
learned query vectors ──────────────────────► Q-Former
                                                   │
text prompt ───────────────────────────────┐       ▼
                                           └─► frozen LLM ──► caption or answer
                                                   ▲
                                      projection of query features
```

The learned queries are like a small team. The team asks: “which visual data must we extract?” The queries use self-attention. They also use cross-attention to the image features. The result is a fixed-size set of visual representations.

Training has two stages:

1. **Vision–language representation learning.** Connect the Q-Former to the frozen image encoder. Train it with image–text contrastive learning, image–text matching, and image-grounded text generation. Different attention masks support these tasks.
2. **Vision-to-language generative learning.** Project the query features of the Q-Former into the input embedding space of the LLM. They act like soft visual prompt tokens. Train the bridge. As a result, the frozen LLM generates text with the image as condition.

The main saving is this. The large backbones need no update. The word “lightweight” compares the trainable bridge with the complete system. It does not guarantee cheap inference. Also, you can unfreeze components for fine-tuning. The frozen-backbone rule describes only the original pre-training strategy.

So, read BLIP-2 as a **modular bridge design**. It is not only a better joint embedding space. Its usual purpose is image-conditioned generation and visual question answering (VQA). But its first-stage components also support retrieval.

## 9. Compare the designs on the correct properties

“Which is best?” does not give enough information. Ask these questions instead. Which interface do I need? What is the training cost? Which task do I have?

| Model | Main architecture | Core idea | Natural uses |
|---|---|---|---|
| **CLIP** | Dual encoders | Symmetric softmax contrastive loss in a joint space | Retrieval, prompt-based zero-shot classification |
| **SigLIP / SigLIP 2** | Dual encoders, with more training methods in SigLIP 2 | Pairwise sigmoid matching. Better multilingual and dense features in v2 | Retrieval, classification, visual representations |
| **BLIP** | Dual-encoder path + image-grounded encoder and decoder | Combine ITC, ITM, and LM. Bootstrap cleaner training data | Retrieval, detailed pair matching, captioning, VQA |
| **BLIP-2** | Frozen vision model + Q-Former + frozen LLM | Train a small bridge, not the large backbones | Image-conditioned generation and VQA. Retrieval with earlier components |

Two systems can have similar embeddings but different training losses. Different systems can have similar names but very different outputs. Compare these properties separately:

- **Architecture:** what processes the inputs? Which weights train?
- **Objective:** which error signal changes the weights?
- **Interface:** does inference give an embedding, a pair score, class scores, or generated text?

A cosine similarity, an ITM logit, a softmax class score, and a generated sentence are different outputs. You cannot exchange them.

### Other useful models

- **[ALIGN](https://arxiv.org/abs/2102.05918):** dual-encoder contrastive learning at large scale. It uses noisy image and alt-text pairs.
- **[CoCa](https://arxiv.org/abs/2205.01917):** combines contrastive learning and caption generation in one pre-training design.
- **[LiT](https://arxiv.org/abs/2111.07991):** locked-image tuning. Keep a pretrained image tower frozen. Train only the text tower with a contrastive loss.
- **[OpenCLIP](https://github.com/mlfoundations/open_clip):** an open-source implementation and a family of reproduced models. It is not one new architecture.

## 10. What can go wrong?

1. **False negatives.** Two captions in a batch can describe the same image. If you give one caption a negative label, the training signal is contradictory. Real datasets need careful checks here.
2. **A shared space is not perfect understanding.** Image embeddings and text embeddings can occupy partially separate regions. This effect is a **modality gap**. Also, a high score can show a correlation in the data. It does not always show the concept that you want.
3. **Similarity is not certainty.** Softmax scores are relative to the candidates that you supply. If you add a different candidate, the probabilities change.
4. **Retrieval evaluation needs held-out data.** Recall@K measures how often the correct partner is in the top K retrieved items. Measure it on data that training did not use. Measure it in two directions.
5. **Pre-training data is important.** Bias, harmful associations, prompt sensitivity, and weak coverage of specialist domains stay. A flexible interface does not remove them.
6. **Dual encoding and joint fusion have different costs.** A search in cached embeddings is scalable. The evaluation of each image–text pair through cross-attention can be much more expensive.

## 11. Exercises

1. **Break a pair.** Swap two captions before training. Which pairing do the diagonal targets then give? Why can the loss go down and the assertions pass, although the printed associations are wrong? Change the order back after the test.
2. **Remove one direction.** Use only `F.cross_entropy(logits, targets)`. Compare the two retrieval directions. What changes? This toy example separates easily. As a result, both directions can stay correct. Use a harder dataset to show the difference.
3. **Change the space.** Set `embedding_dim=2` and print the embeddings. Is the small space sufficient for this toy example? Why is it wrong to read its axes as named concepts?
4. **Test generalisation honestly.** Keep one colour–caption pair out of *all* training steps. For example, train on `images[1:]` and `text_inputs[1:]`. Then test with all four images and all four captions. Describe the result. Do not take one run as proof.
5. **Implement SigLIP.** Put `siglip_loss` below the imports in `tiny_clip.py`. Use it instead of the CLIP loss. Add a learned scalar bias to the model. Add the bias to the optimiser. Compare the similarity matrices. This small run cannot show the distributed-training advantages of SigLIP.
6. **Use an unknown word.** Try to encode `a photo of a purple square`. Why does `bag_of_words` cause a `KeyError` for `purple`? What does this result show about the toy vocabulary?
7. **Select an architecture.** For each task, give a reason for your starting point. Task 1: search one million images with text. Task 2: write a paragraph about an image. Task 3: answer a question about a scan with an existing frozen LLM. Task 4: classify images without task-specific labels.

## 12. The one-paragraph mental model

CLIP and SigLIP train separate image encoders and text encoders to agree in a shared vector space. CLIP contrasts each example with the batch alternatives by softmax. SigLIP uses pairwise sigmoid decisions. BLIP adds joint image-grounded matching and text generation. It also has a method to clean noisy training captions. BLIP-2 uses a different method. It keeps large pretrained vision and language models frozen. It trains a Q-Former bridge between them. The embedding space is a good first idea. But the **objective, the architecture, and the inference interface** show which model is correct for your project.

### Primary sources

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP). See the pseudocode in Figure 3.
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (SigLIP).
- Tschannen et al., [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786).
- Li et al., [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086).
- Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597).
