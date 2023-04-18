import pandas as pd
import numpy as np
import jax.numpy as jnp

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

if __name__=="__MAIN__":
    one_hot()
        