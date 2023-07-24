import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FluxMaskLayer(layers.Layer):
  """
  A layer that incorporates the information on flux signal (its use is optional)
  """
  def __init__(self, flux_signals, **kwargs):
    super(FluxMaskLayer, self).__init__(**kwargs)
    self.flux_mask = tf.constant(flux_signals, dtype=tf.float32)
  
  def get_config(self):
        config = super().get_config()
        config.update({"flux_signals": self.flux_mask.numpy().tolist()})
        return config

  def call(self, inputs):
    return tf.abs(inputs) * self.flux_mask


class NE_Loss(tf.keras.losses.Loss):
    """
    Normalized Error Loss
    """
    def __init__(self,reduction='auto'):
        super().__init__()
        self.reduction = reduction
    def call(self, y_true, y_pred):     
        loss = tf.norm(y_true - y_pred, axis=1) / tf.norm(y_true, axis=1)
        return tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)


class HyperModel(keras_tuner.HyperModel):
  """
  HyperModel class that defines the search space for our deep learning architectures and respective hyperoptimization
  """
  def __init__(self, output_shape, flux_mask):
    super(HyperModel, self).__init__()
    self.output_shape = output_shape
    self.flux_mask = flux_mask

  def build(self, hp):
    model = keras.Sequential()

    activation = hp.Choice("activation", ["relu",
                                          "tanh",
                                          "elu",
                                          "linear",
                                          "selu",
                                          "sigmoid",
                                          "softmax",
                                          "swish"])

    regularizer = hp.Choice("regularizer", ["l1", "l2"])

    for i in range(hp.Int('num_layers', 1, 5)):
      model.add(layers.Dense(units=hp.Int('units_' + str(i),
                              min_value=5,
                              max_value=50,
                              step=5),
                              activation=activation,
                              kernel_regularizer=regularizer))
    if hp.Boolean("dropout"):
          model.add(layers.Dropout(rate=hp.Float('dropout_rate',
                                    min_value=0.1,
                                    max_value=0.5,
                                    sampling="linear")))

    model.add(layers.Dense(self.output_shape, kernel_regularizer=regularizer, activation="relu"))

    if self.flux_mask:
      model.add(FluxMaskLayer(self.flux_mask))

    learning_rate = hp.Float("lr", min_value=1e-3, max_value=1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss = NE_Loss())
    
    return model