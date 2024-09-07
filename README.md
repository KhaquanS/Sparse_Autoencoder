# Overview
<img src="https://github.com/user-attachments/assets/216df41a-dc2e-4c89-be67-97e1359157e3" alt="SAE" width="400" height="400"/></br>
This repo presents a very simplistic implementation of the Sparse Auto-Encoder (SAE). The SAE are a class of regularised Auto-Encoders which provide the following major benefit:
* SAEs introdue a sparsity contraint on the latent space representation of the input data learnt by the encoder. This contraint mainly prevents the latent space from being a naive 'copy' of the input data. The benefits of this contraint are largely felt in the over-complete case where the dimensionality of the latent space is **larger** than that of the input, however, it also might be helpful in the under-complete case where the dimensionality of the latent space is **smaller** than that of the input.

 
