

from tf_train.trainer import Trainer
from cnn_2 import cnn_archi


import config

trainer = Trainer( cnn_archi, config )

trainer.train()