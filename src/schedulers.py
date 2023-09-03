import tensorflow.keras as tfk

class AdaptiveSchedulerCallback(tfk.callbacks.Callback):
  """An adaptive learning rate scheduler in which the learning rate increases/decreases with respect to training loss.
  if the loss gets smaller in an epoch the learning rate will be increased.
  if the loss gets bigger in an epoch the learning rate will be decreased.

  :param k: a coefficient which will be multiplied to lr if the loss gets smaller. (k > 1)
  """

  def __init__(self, k: int = 1.1):
    if k < 1:
      raise ValueError('Parameter k must be bigger than 1!')
    super(AdaptiveSchedulerCallback, self).__init__()
    self.last_loss = float('inf')
    self.k = k

  def on_epoch_end(self, epoch, logs = None):
    # Get the current learning rate from model's optimizer.
    lr = float(tfk.backend.get_value(self.model.optimizer.learning_rate))
    # get current loss
    current_loss = logs['val_loss']
    # implement the scheduler
    if current_loss < self.last_loss:
      scheduled_lr = lr * self.k
    else:
      scheduled_lr = lr * (2 - self.k)
    # update lr
    tfk.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    # update last_loss
    self.last_loss = current_loss

def step_decay_scheduler(epoch: int, lr: int, alpha0: float = 0.001, k: float = 0.5, thresh: int = 20):
    '''
    Calculates the next learning rate value formula: lr = alpha0 * k ^ (epoch // thresh)
    :param epoch: index of the current epoch starting from 0
    :param lr: current learning rate value
    :param alpha0: the initial learning rate
    :param k: decay coefficient. the lr value devides by this number every $num_epochs epochs.
    :param num_epochs: number of consecutive epochs in which the lr is constant.
    '''
    return alpha0 * (k ** (epoch // thresh))
