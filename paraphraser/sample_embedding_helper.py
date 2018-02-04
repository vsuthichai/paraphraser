from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.helper import GreedyEmbeddingHelper


class MySampleEmbeddingHelper(GreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_tokens, end_token,
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(MySampleEmbeddingHelper, self).__init__(
        embedding, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    if self._softmax_temperature is None:
      logits = outputs
    else:
      #logits = outputs / self._softmax_temperature
      logits = math_ops.divide(outputs, self._softmax_temperature)

    sample_id_sampler = categorical.Categorical(logits=logits)
    sample_ids = sample_id_sampler.sample(seed=self._seed)

    return sample_ids

