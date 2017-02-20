## TODO:

- [ ] **Discuss and verify whether the batch gradient update implementation is correct.**

- [ ] Check if the 3K labeled-training and 30K unlabeled-training contains the same distribution of digits.

- [ ] Implement SWWAE.

- [ ] Ensure that you give predictions for all the test cases because some cases might be missed due to batching. Pad with some dummy cases to avoid this.

- [ ] Check if the labels are correct for the corresponding images.

- [x] Extend Denoising Autoencoder to Stacked Denoising Autoencoder.

- [x] Add shuffling step in CNN training.

- [ ] Implement Image Whitening
    - http://ufldl.stanford.edu/tutorial/unsupervised/ExercisePCAWhitening/

- [ ] Ladder Network

- [ ] SdA Into Deep Conv Net



## Random Alex Ideas

- Variable Dropout Rates

- Examine Model Prediction Confidence
    - Consider additional actions on low confidence predictions

## YASU ADVICE
Not sure it works on MNist since black/white
Step 1: patch unlabeled images
Patch size is same as filters in the first layer of your network
Step 2: run kmeans on the patches
K is the number of filters in your first layer
Step 3: take centroids out
    
## References

- Deconstructing the Ladder Network Architecture [[paper]](https://arxiv.org/abs/1511.06430).

- Ladder Networks [[article]](http://rinuboney.github.io/2016/01/19/ladder-network.html).

- Ladder Networks implementations: [[original]](https://github.com/CuriousAI/ladder), [[tensorflow]](https://github.com/rinuboney/ladder).

- Code for reproducing results of NIPS 2014 paper "Semi-Supervised Learning with Deep Generative Models" [[repo]](https://github.com/dpkingma/nips14-ssl).

- VAE example in PyTorch [[code]](https://github.com/pytorch/examples/blob/master/vae/main.py).
