# State of Analysis (SoA) of the project

We started the Homeostatic RL project in order to analyze the homeostatic behavior and neural activity of agents and compare that with the neural activity of the brain. We cloned and managed to run the basic two resource env where the agent has to maintain two internal states or resource levels (blue and red) within a certain range. The agent has to learn certain behaviors in order to maintain these levels and ultimately survive. The agent receives feedback on its internal state as a component of its reward. Certain interesting behaviors emerge based on the internal state. In this project we are focusing on the foraging behavior of the agent. 

## Experiments performed
- <b>Characterizing the Agent and Environment:</b> 
    The network has three inputs along with the following dimensions:
    - Exteroception (or the input of external state): 40 dim
    - Proprioception (or the input of body's movement or position in space): 27 dim
    - Interoception (or the input of the internal state of the body): 2 dim

    The entire state space and action space is explained in detail here. <a href="https://github.com/CMC-lab/hrd/blob/01535ec66d591563f56759b2782e27f24fba1543/notes/network_characterization.md">link</a>

- <b>Connectivity analysis of the trained weights:</b> We plotted the connectivity matrix of multiple neurons as a heatmap to observe any patterns in the trained weights. All the plots were plotted and can be found here <a href="https://github.com/CMC-lab/hrd/tree/01535ec66d591563f56759b2782e27f24fba1543/src/hrd/hrl_bs_ijcnn2023/plots/connectivity_matrix">link</a>.
- <b>Visual analysis of the agent's Behavior:</b> 
    <!-- - We observed a binary behavior where the agent moves around and eats when it's hungry else it doesn't move at all.
    - When the agent is very hungry with internal state (-0.7, -0.7), it doesn't move at all because moving also costs energy. -->
    - We ran simulations with different initial internal states and observed the behavior of the agent. The agent had different behavior in the environment based on the initial internal state and the distance to the resource. The value of internal state seems to affect the speed of the agent as well. More details regarding this can be seen in the observations section.

- <b>Neural activity analysis:</b> 
## Observations
- Connectivity analysis of the trained weights: 
    
- Visual analysis of the agent's Behavior: 

- Neural activity analysis:
 

## Results
#refer generated images and their codes

## Next steps
