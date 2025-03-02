# Code for the Inference Time Compute Hackathon

## Steps to use:

1. Run `bash launch_llms.sh``

Fair warning: you will need ~8 H100s of compute to store the main model as well as the verifier model

2. Run `python model.py`

This will launch our Agentic GoT model and you will be able to converse with it. Sample reasoning traces can be visualized at v-kaarti.github.io.

### Steps to reproduce:

`datasets/` contain the logical fallacy datasets useful for training logical fallacy verifier models
`experiments/` contains all the notebooks used to generate our major results.

