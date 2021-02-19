# DNNs for ASVspoof
## About 
I believe that open cluture let a community bigger like Linux.
So I create this project for ASVspoof community to share the DNNs implementation.
Please contribute your models to this projects!
I hope this project can help more people.

All models are implemented as tf.keras.models.Model class.
Therefore you can reuse them in your personal project easily.

***Notice: I'm looking for partners to manage this project.*** 
***If you're interested, please contact me.***

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

Tensorflow >= 2.0

Keras

Numpy (for unit testing)

## Contribution
Contribution is more than welcome.

To add your model to my project, you should firstly run 

```python new_model.py MODEL_NAME``` 

Then your model directory should be automatically created with template files.

***The models should be implemented as a tensorflow.keras.models.Model class.***

**Please don't forget test your model and write description before pull request.**


## Appriciate 
I really appriciate to the studies which I refered for creating this repository and all of my contributor.


## Reference
[ASVspoof2019 Challenge](https://www.asvspoof.org)
