What Makes A Good Story? Designing Composite Rewards for Visual Storytelling
===
![ReCo-RL Model](https://github.com/JunjieHu/ReCo-RL/blob/master/model.png)
Implemented by [Junjie Hu](http://www.cs.cmu.edu/~junjieh/)

Contact: junjieh@cs.cmu.edu

If you use the codes in this repo, please cite our [AAAI2020 paper](https://arxiv.org/abs/1909.05316).

	@inproceedings{hu20aaai,
	    title = {What Makes A Good Story? Designing Composite Rewards for Visual Storytelling},
	    author = {Junjie Hu and Yu Cheng and Zhe Gan and Jingjing Liu and Jianfeng Gao and Graham Neubig},
	    booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
	    address = {New York, USA},
	    month = {February},
	    url = {https://arxiv.org/abs/1909.05316},
	    year = {2020}
	}

Installation
==
Please use the following Anaconda environment.
- python=3.6
- pytorch=1.0.1
- pytorch_pretrained_bert=0.6.2
- spacy, nltk, numpy, scipy, h5py, json, pickle

```
conda env create --file conda-env.txt
```


Downloads
==
The preprocessed data and pre-trained models can be found [here](https://drive.google.com/drive/folders/1j8P6CwykJDAIV6Et7bci_x00mvBAwmVm?usp=sharing). Extract ***data.zip*** under the ***ReCo-RL/data*** directory. Extract ***reco-rl-model.zip*** under the ***ReCo-RL/outputs/rl/*** directory. Extract ***bert-base-uncased.zip*** under ***ReCo-RL/bert-weight/*** directory.

- ***data.zip***: train/dev/test data including image features, VIST captions and entities preprocessed by spacy.
- ***reco-rl-model.zip***: model file (model.bin) and vocab file (vocab.bin).
- ***bert-base-uncased.zip***: BERT's next sentence predictor model and its vocab file.


Demo
==
- Decode the test set using pre-trained model. We recommend to use our pre-trained model for further comparison in your paper. After decoding, you would expect to see the decoding automatic scores as follows (improved scores over those reported in the paper).

|Score|SPICE|BLEU-4|ROUGE-L|CIDEr|METEOR|
|---|---|---|---|---|---|
|ReCo-RL|11.2|13.5|29.5|11.8|35.8| 

```
bash scripts/test.sh [GPU id]
```

- Train a MLE model from scratch
```
bash scripts/train_mle.sh [GPU id]
```

- Train the model initialized by the MLE-trained model
```
bash scripts/train_rl.sh [GPU id]
```
