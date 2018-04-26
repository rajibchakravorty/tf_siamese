

from tf_train.trainer import Trainer


import sample_1000.config as config

trainer = Trainer( config.network, config )

trainer.train()
