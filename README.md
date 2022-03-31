# IANNwtf_Final_Project

Repository of the Final project "Generating Optionated Movie Results using GANs" by Lara McDonald and Thomas Nortmann in the course "Implementing ANNs with TensorFlow" in the WS 2021/22 

## run.py
Script that trains both Models and uses the trained models to generate movie reviews afterwards

## vae.py
The class resembling the Variational Autoencoder

## generator.py
The class of the Generator of the Generative Adversarial Network

## discriminator.py
The class of the Discriminator of the Generative Adversarial Network

## training_loop.py
The class holding the logic how to train and tests the VAE and the GAN

## input_pipeline.py
The class preparing the dataset for the models

## vocab.txt
The vocabulatory used by the Tokenizer in input_pipeline.py

## saved models
The weights of both the VAE and GAN, can be used by setting train_gan and train_vae to false in run.py
