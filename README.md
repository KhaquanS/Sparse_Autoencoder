# Overview
<div align="center">
    <img src="https://github.com/user-attachments/assets/367952a0-dbb4-4f39-a689-7e9d551770f7" alt="SAE inference sample" width="400" height="400"/>
    <p><em>Figure: Original images on the top with their SAE reconstructed versions at the bottom.</em></p>
</div>

This repo presents a very simplistic implementation of the Sparse Auto-Encoder (SAE). The SAE are a class of regularised Auto-Encoders which provide the following major benefit:
* SAEs introdue a sparsity contraint on the latent space representation of the input data learnt by the encoder. This contraint mainly prevents the latent space from being a naive 'copy' of the input data. The benefits of this contraint are largely felt in the over-complete case where the dimensionality of the latent space is **larger** than that of the input, however, it also might be helpful in the under-complete case where the dimensionality of the latent space is **smaller** than that of the input.

# Using the Repo 
You can clone the repo to your local workspace and use the following terminal commands to train your own SAE on the MNIST data:
1. cd SAE
2. !sae.py

Furthermore, you can pass in the following arguments at the terminal to further customise the model/training procedure:
* batch_size -> int: Sets the batch size for training the model -- default = 64.
* epochs -> int: Sets the number of epochs for training -- default = 20.
* lr -> float: Sets the learning rate -- default = 1e-4.
* in_dims -> int: Sets input dimensions of the model -- default = 784 (MNIST size).
* h_dims -> str: Sets hidden dimensions of the model --default = \[20\].
* sparsity_lambda -> float: Sets the lambda value for the SAE, which controls the strength of the sparsity penalty in the loss function. A larger value results in more sparsity but potential loss in reconstruction quality -- default = 0.2.
* sparsity_target -> float: Sets the sparsity target value, which is the average activation you want each hidden neuron to have across all input samples. A smaller value allows more sparsity -- default = 0.01.
* download_mnist -> bool: Downloads the MNIST dataset for training -- default = True.
* train -> bool: Launches training -- default = True.
* save_model -> bool: Saves the model to a folder called 'files' -- default = False.
* num_samples -> int: Specifies the number of samples to visualize -- default = 5.
* save_plots -> bool: Saves the sample plots and the training loss plots to a folder called 'images' -- default = True.
