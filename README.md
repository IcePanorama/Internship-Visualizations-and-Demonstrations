# Internship Visualizations and Demonstrations 

This repository is a collection of some of the programs that I've written over the course of my internship at the Department of Homeland Security's Center for Accelerating Opperational Effeciency. All of the following programs were entirely programmed by myself.

## Accuracy At Varying ε

`AccuracyPerEpsilonValue.py` demonstrates how a tighter privacy budget (i.e., a lower epsilon value) in a differentially private logistic regresion model results in a lower accuracy than similar models with less strict or even no privacy applied at all.

This program was written using the `numpy`, `pandas`, `scikit-learn`, and `matplotlib` libraries.

## Dummy Membership Inference Attack

`MIA_Simulation.py` simulates a membership inference attack (MIA), which exploits machine learning model output probabilities to infer whether or not someone’s data was used in the creation of a machine learning model, on a logistic regression model and calculates the attack accuracy.

This program was created for educational purposes in order to educate audiences on how a MIA works.  Unlike a normal MIA, this program doesn't use any synthetic data, due to the limited technology available to us during the time of our internship.

Because of these restrictions, this MIA model simply splits an existing dataset in half, using one half for training the original model and using the second half for training the adversarial/shadow model.

The model works as follows: first a regular logistic regression model is trained using one half of the dataset.  Next, the `x` values of the second half of the dataset are used by the original model in order to make predictions which are then stored in `y_pred`. Finally, using both `x` and `y_pred` together, we can train an adversarial/shadow model. Using this new adversarial/shadow model, if we feed in `x` data from the original training dataset, we can predict whether or not that piece of data was used in the training of the original model roughly 90% of the time.

This program was written using the `numpy`, `pandas`, and `scikit-learn` libraries.

## Membership Inference Attack

A [membership inference attack (MIA)](https://arxiv.org/abs/1610.05820) exploits machine learning (ML) model output probabilities to infer whether or not someone’s data was used in the creation of said ML model.

This program creates two different MIAs, one which works as if an attacker had access to just a single row of the original dataset and another which works as if an attacker had access to a small subset of the original dataset. Unlike [Dummy Membership Inference Attack](#dummy-membership-attack), this program uses synthetic data (created using a generative adversarial network) as part of its attack on a model. Both shadow models are then tested against the entire original dataset to measure their accuracy.

This program was written using `PyTorch` along with the `numpy`, `pandas`, and `scikit-learn` libraries.


## Visualizations Generator

`Visualizations.py` generated various assets which I then used in GIMP, a photo editing software, to create preliminary visualizations for our team's research.

This program was written using the `numpy` and `matplotlib` libraries.
