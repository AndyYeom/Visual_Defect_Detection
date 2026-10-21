# Architecture and decisions

The CLI and Streamlit app share the Detector class. Training serializes both model state dictionaries, normal feature statistics and all score calibration into a single checkpoint. Loading uses PyTorch's weights-only mode and reconstructs the CNN without a network download. Use checkpoints produced by the team.

## Methods

The autoencoder has three downsampling convolutions and three transposed-convolution decoding stages. It learns RGB reconstruction with mean squared error. Its map is mean squared channel error.

The feature baseline freezes ImageNet ResNet18 through layer2, normalizes inputs with the documented ImageNet mean/std, and estimates one mean and diagonal variance per spatial location/channel from training images. A variance floor avoids division by zero. Distance averages squared standardized feature deviations across channels; bilinear interpolation returns an input-resolution map. This is a simple Gaussian feature baseline, not full-covariance PaDiM.

For each method, the average of the top 1% of map values produces the image score. Normal-validation mean/std standardization precedes weighted score averaging. Each baseline and the ensemble receives its own normal-validation quantile threshold. No scores are clipped to probabilities.

## Tradeoffs

CPU-first execution and 128-pixel images make setup accessible. The frozen spatial feature model needs reasonably aligned images and may fail under pose or lighting changes. Small defects may disappear under resize. Variance estimates need enough normal images. GPU training, richer architectures, checkpoint resumption, pixel metrics and multi-user hosting are future work.

Streamlit is a local single-user demo. It writes the latest overlay and an append-only local review log. It is not designed as a shared production service.
