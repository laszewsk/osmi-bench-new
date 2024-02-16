# some helper libraries for the tutorial
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import numpy as np

# import smartsim and smartredis
from smartredis import Client
from smartsim import Experiment

exp = Experiment("Inference-Tutorial", launcher="local")

db = exp.create_database(port=6780, interface="lo")
exp.start(db)
