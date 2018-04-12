

from tf_train.trainer import Trainer


import sample_100.config as config

trainer = Trainer( config.network, config )

trainer.train()
