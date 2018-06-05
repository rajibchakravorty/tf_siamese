

from tf_train.trainer import Trainer


#import sample_1000.config_gouk as config
import sample_1000.config_hoffer as config
#import sample_1000.config_hsu as config
#import sample_1000.config_lewis as config

trainer = Trainer( config.network, config )

trainer.train()
