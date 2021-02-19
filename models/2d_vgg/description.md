# 2 Dimensional VGG

2 Dimentional VGG, which is identified by two separated inputs for two different spectrograms, is proposed in ASVspoof2019 Challenge.


## Reference
***[Detecting Spoofing Attacks Using VGG and SincNet: BUT-Omilia Submission to ASVspoof 2019 Challenge](https://arxiv.org/pdf/1907.12908.pdf)***

[ Hossein Zeinali, Themos Stafylakis, Georgia Athanasopoulou, Johan Rohdin, Ioannis Gkinis, Lukáš Burget, Jan "Honza'' Černocký ]


## Description

2 Dimensional VGG comprises several convolutional and pooling layers followed by a statistics pooling and several dense layers which perform classification.  There are 6 convolutional blocks in the model, each containing 2 convolutional layers and one max-pooling. Each max-pooling layer reduce the size of frequency axis to half while only one of them reduces the temporal resolution.
After the convolutional layers, there is a mean pooling layer which operates only on the time axis and calculates the mean over time. After this layer, there is a flatten layer which simply concatenates the 4 remaining frequency channels. Finally there are 3 dense layers which perform the classification task.