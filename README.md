# ThymioSwarm
Real hardware evolutionary design pipeline for heterogeneous robot swarms.

------
#### This codebase was used in the following paper:
### Learning online adaptive emergent control for heterogeneous robot swarms

Link to this work can be found here

This repository covers both the **simulation** and **hardware** experiments presented in the paper.

* To reproduce the simulation experiment as presented in the paper, follow the instructions [here](#reproduce-simulation-experiments):
* To reproduce the hardware experiment as presented in the paper, follow the instructions [here](#reproduce-hardware-experiments):
* To obtain the data and reproduce our analysis results as presented in the paper, follow the instructions [here](#reproduce-results):



### Citation:
```
@article{van2024learning,
  title={ Learning online adaptive emergent control for heterogeneous robot swarms},
  author={van Diggelen, Fuda and Alperen Karagüzel, Tugay and Rincón, Andrés García and Ferrante, Eliseo},
  year={2024}
}
```

Publications
------
#### This repo is directly related to the following papers:
* Van Diggelen, F., Luo, J., Karagüzel, T. A., Cambier, N., Ferrante, E., & Eiben, A. E. (2022, July). Environment induced emergence of collective behavior in evolving swarms with limited sensing. In _Proceedings of the Genetic and Evolutionary Computation Conference_ (pp. 31-39). https://doi.org/10.1145/3512290.3528735. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/GECCO_2022)
* Van Diggelen, F., De Carlo, M., Cambier, N., Ferrante, E., & Eiben, A. E. (2024). Emergence of specialized Collective Behaviors in Evolving Heterogeneous Swarms. _Arxiv_: https://arxiv.org/abs/2402.04763. [**[Branch]**](https://github.com/fudavd/EC_swarm/tree/GECCO_2024)

---------------
## Reproduce simulation experiments
- Set up the EC_swarm sim_exp environment, from https://github.com/fudavd/EC_swarm/tree/CrazyThymio
```bash
git submodule init
git submodule update
cd ./sim_exp/ 
```
Follow the installation instruction [here](https://github.com/fudavd/EC_swarm/tree/CrazyThymio?tab=readme-ov-file#installation)

To run the Evolutionary experiment for the _Baseline controller_
```bash
./run-experiment.sh Baseline_swarm_EvoExp
```
To run the Evolutionary experiment for the _Heterogeneous controller_
```bash
./run-experiment.sh Hetero_swarm_EvoExp
```
To run the ratio experiment for the _Adaptive Heterogeneous controller_
```bash
cd ./sim_exp/
wget https://dataverse.nl/dataset.xhtml?persistentId=doi%3A10.34894%2FBSZNMT#
unzip ./results.zip
python ./sim_exp/results/RetestBest.py
```
--------------
## Reproduce hardware experiments
- Prepare a clean Raspberry Pi install, to connect with the Thymio using an [imager](https://www.raspberrypi.com/software/)
- Upload and run the `./real_exp/initial_config.sh` on the Pi
```bash
scp ./real_exp/initial_config.sh <user>@pi_ip_address:~/Desktop/
ssh <user>@pi_ip_address 'chmod +x ~/Desktop/initial_config.sh'
ssh <user>@pi_ip_address '/bin/bash/ ~/Desktop/initial_config.sh'
```

- On the sensor hardware, install the correct firmware provided in `real_exp/CrazyThymio-firmware/examples/app_share_pos`


- After initial configuration we can run a controller (example baseline) after establishing an ssh connection with the following commands:
```bash
cd ~/Desktop/crazy_thymio/CrazyThymio/
source ../.venv/bin/activate 
python thymio_baseline.py
```
-------
## Reproduce results
Replication data can be downloaded from here https://doi.org/10.34894/2FBSZNMT
To automaticcaly download and run:
```bash
wget https://dataverse.nl/dataset.xhtml?persistentId=doi%3A10.34894%2FBSZNMT#
unzip ./results.zip
```
For the simulation data analysis:
`python ./results/sim_exp/Analize_swarm.py`

For the noise model data analysis:
`python ./results/noise_model/Model_noise.py`

