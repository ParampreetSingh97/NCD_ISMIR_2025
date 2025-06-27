import tensorflow as tf
import numpy as np
class MCDropoutModel(tf.keras.Model):
    def __init__(self, base_model):
        super(MCDropoutModel, self).__init__()
        self.base_model = base_model

    def call(self, inputs, training=True):  # Always forces dropout active
        return self.base_model(inputs, training=True)

import numpy as np

# Function to apply MC Dropout for confidence estimation on multiple samples
def mc_dropout_predictions(mc_model, x_samples, num_samples=50):
    """
    Perform MC Dropout inference on multiple samples.
    
    Args:
        mc_model: The Keras model with Dropout enabled during inference.
        x_samples: The input samples of shape (batch_size, t, f, num).
        num_samples: Number of stochastic forward passes.
    
    Returns:
        preds: An array of shape (num_samples, batch_size, num_classes),
               containing multiple predictions for each sample.
    """
    preds = np.array([mc_model(x_samples, training=True).numpy() for _ in range(num_samples)])
    return preds  # Shape: (num_samples, batch_size, num_classes)

def batchwise_mc_dropout(mc_model, x_samples, num_samples=50, n_batch=5):
    """
    Compute MC Dropout predictions in batches.
    
    Parameters:
    - mc_model: The model with MC Dropout enabled.
    - x_samples: The input data (shape: (N, t, f, num)).
    - num_samples: Number of stochastic forward passes.
    - n_batch: Number of samples per batch.

    Returns:
    - Stacked MC Dropout predictions for all samples.
    """
    num_total = x_samples.shape[0]  # Total number of samples
    mc_predictions = []

    for i in range(0, num_total, n_batch):
        x_batch = x_samples[i : i + n_batch]  # Extract batch
        mc_batch_preds = mc_dropout_predictions(mc_model, x_batch, num_samples)  # MC Dropout
        mc_predictions.append(mc_batch_preds)  # Store batch results
        print("Batch", i)

    # Stack all batch results into a single array
    return mc_predictions

