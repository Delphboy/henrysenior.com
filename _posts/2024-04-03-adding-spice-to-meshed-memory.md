---
title: Adding SPICE to Meshed-Memory Transformer
collection: words
permalink: /words/2024-04-03-adding-spice-to-meshed-memory
excerpt: "A quick guide to adding the SPICE evaluation metric to the meshed-memory transformer codebase"
---

## SPICE Evaluation Support
### File Setup
1. Download the ZIP file for the [COCO evals repo](https://github.com/ruotianluo/coco-caption)
2. Extract the zip folder
3. Copy the `pycocoevalcap/spice/` directory into the `evaluation/` directory of Meshed Memory directory
4. Copy the `get_stanford_models.sh` bash script into the `evaluation/` directory

### File Edits
1. Edit the `get_stanford_models.sh` so the `SPICELIB` variable is as follows: `SPICELIB=spice/lib`
2. Add `from .spice import Spice` to the top of `evaluation/__init__.py`
3. Add `from .spice import Spice` to `evaluation/spice/__init__.py`

### Run
1. Run the `get_stanford_models.sh` from within the `evalations/` directory

### Additional Edits
#### Pretty Print the SPICE Metric
The `compute_scores` method in `evaluation/__init__` uses the `str()` function to get the metric name. To prevent the function printing a memory address, edit `evaluation/spice.py` and change:
```python
def method(self):
        return "SPICE"
```
to
```python
def __str__(self):
        return "SPICE"
```

#### Control Whether SPICE is Used
SPICE can be quite slow to run, so you may not want to run it after every training epoch on the validation set, instead opting to run it only on the test set after training is complete. In order to do this, edit the `compute_scores` function in `evaluation/__init__.py` to be the following:

```python
def compute_scores(gts, gen, is_test=False):
    if is_test:
        metrics = (Bleu(), Meteor(), Rouge(), Cider(), Spice())
    else:
        metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores
```

Then ensure you add `is_test=True` to the `compute_scores` call you use for your test set evaluation.
