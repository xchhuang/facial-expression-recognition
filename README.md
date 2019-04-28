# Facial Expression Recognition

This is the implementation of part of my undergraduate thesis "Feature-Level Joint Learning for Facial Expression Recognition". **Notice**: I am still refactoring the code in Keras2.

<!-- and the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8528894) "Facial Expression Recognition with Identity and
Emotion Joint Learning".  -->

### Prerequisites
* Python 3.6.5 | Anaconda
* Keras 2.1.6

### Pipeline

#### Data
* Download the FER+ dataset in [google drive](https://drive.google.com/file/d/1zzd8uc5bau0gVmK9iQ01dqC_3S0IUYoq/view?usp=sharing) and put it into the data folder.

#### Run
* Run `python main.py --aug=False` to train and evalute our model, with no data augmentation. You can also play with the parameters in `params.py` based on the paper (e.g., batch size).


<!-- ### Acknowledgement

If you find this repository useful, please cite our paper:
```
@inproceedings{huang2017cross,
  title={Cross-domain sentiment classification via topic-related TrAdaBoost},
  author={Huang, Xingchang and Rao, Yanghui and Xie, Haoran and Wong, Tak-Lam and Wang, Fu Lee},
  booktitle={Thirty-First AAAI Conference on Artificial Intelligence},
  year={2017}
}
``` -->


