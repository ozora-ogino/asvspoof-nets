# DNNs for ASVspoof

## About 

All models are implemented as tf.keras.models.Model class.
Therefore you can reuse them in your personal project.

## Description
### Usage

```git clone https://github.com/ozora-ogino/asvspoof-nets```

Then import models to your code!!


### File Explanation

models/*MODEL_NAME*/models.py : Models are implemented as tf.keras.models.Model class.

models/*MODEL_NAME*/layers.py : There are custom layers. (optional)

models/*MODEL_NAME*/test_models.py : Unittest

models/*MODEL_NAME*/description.md : Description and reference.

### Requirements

Tensorflow >= 2.0.0 

Keras

Numpy (for unit test)

## Contribution
Contribution is more than welcome.

To contribute, you should run 

```python new_model.py MODEL_NAME``` 

Then your directory should be automatically created with template files.

**Please don't forget test your model and write description before pull request.**


## Appriciate 
I really appriciate to the studies which I refered for creating this repository and all of my contributor.


## Reference
[ASVspoof2019 Challenge](https://www.asvspoof.org)
