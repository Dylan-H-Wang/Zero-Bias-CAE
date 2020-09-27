from trainer import *
from parameters import Parameters

config = Parameters(dataset="covid-ct", data_path="../data/COVID-CT",)

trainer = CAETrainer(config)
trainer.train()
trainer.test()
