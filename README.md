# OSMI-Bench

The Open Surrogate Model Inference (OSMI) Benchmark is a distributed inference benchmark
for analyzing the performance of machine-learned surrogate models and is described in its original form as "smiBench" in the following paper:

> Wesley Brewer, Daniel Martinez, Mathew Boyer, Dylan Jude, Andy Wissink, Ben Parsons, Junqi Yin, and Valentine Anantharaj. "Production Deployment of Machine-Learned Rotorcraft Surrogate Models on HPC." In 2021 IEEE/ACM Workshop on Machine Learning in High Performance Computing Environments (MLHPC), pp. 21-32. IEEE, 2021.

Available from https://ieeexplore.ieee.org/abstract/document/9652868. Note that OSMI-Bench differs from SMI-Bench described in the paper only in that the models that are used in OSMI are trained on synthetic data, whereas the models in SMI were trained using data from proprietary CFD simulations. Also, the OSMI medium and large models are very similar architectures as the SMI medium and large models, but not identical.

OSMI-Bench now supports two types of benchmark tests described here:

| AI framework | inference server | client API    | protocol | 
|--------------|------------------|---------------|----------|
| TensorFlow   | TF Serving       | TF Serving API| gRPC     |
| PyTorch      | RedisAI          | SmartRedis    | RESP     |

Look at the README.md in either tfserving or smartsim to see how to setup and run either benchmark.

