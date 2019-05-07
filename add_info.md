## Additonal Information

#### Interpretability Metric
The scores were computed using the method proposed by [Adel et al.](http://proceedings.mlr.press/v80/adel18a.html). The computation steps are:
1. All data-points in the held-out test set are passed through the encoder of the trained model to obtain the corresponding latent vectors.
2. For each attribute `a`, the latent space dimension `r` which has the maximum mutual information with `a` is computed.
3. A simple linear regression model is then fit to predict `a` given `z_r`, i.e. the value of the latent code for dimension `r`.
4. The interpretability metric is finally the regression score (coefficienct of determination R2) for this regression model

The results are shown in the table below: 


#### Example Generations
Additional examples generated on varying the latent code for the regularized dimensions are provide below

