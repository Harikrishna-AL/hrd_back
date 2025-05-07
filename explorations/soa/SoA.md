# State of Analysis (SoA) of the project

This project aims to analyze the homeostatic behavior and neural activity of agents and compare that with the neural activity of the brain. We cloned and managed to run the basic two-resource env where the agent has to maintain two internal states or resource levels (blue and red) within a certain range. The agent has to learn certain behaviors in order to maintain these levels and ultimately survive. The agent receives feedback on its internal state as a component of its reward. Certain interesting behaviors emerge based on the internal state. In this project we are focusing on the foraging behavior of the agent. 

## Experiments performed
- <b>Characterizing the Agent and Environment:</b> 
    The network has three inputs along with the following dimensions:
    - Exteroception (or the input of external state): 40 dim
    - Proprioception (or the input of body's movement or position in space): 27 dim
    - Interoception (or the input of the internal state of the body): 2 dim

    The entire state space and action space is explained in detail here. <a href="https://github.com/CMC-lab/hrd/blob/01535ec66d591563f56759b2782e27f24fba1543/notes/network_characterization.md">link</a>

- <b>Connectivity analysis of the trained weights:</b> We plotted the connectivity matrix of multiple neurons as a heatmap to observe any patterns in the trained weights. All the plots were plotted and can be found here <a href="https://github.com/CMC-lab/hrd/tree/01535ec66d591563f56759b2782e27f24fba1543/src/hrd/hrl_bs_ijcnn2023/plots/connectivity_matrix">link</a>.
- <b>Visual analysis of the agent's Behavior:</b> 
    - We ran simulations with different initial internal states and observed the behavior of the agent. The agent had different behavior in the environment based on the initial internal state and the distance to the resource. The value of internal state seems to affect the speed of the agent as well. More details regarding this can be seen in the observations section.

- <b>Neural activity analysis:</b> 
    - We plotted the neural activity of both the actor and critic networks. The activity as was dimensionally reduced using PCA to 30 dims and then plotted as a heatmap. The neural activity was plotted for different internal states to spot any patterns in the neural activity. The plots can be found here <a href="https://github.com/CMC-lab/hrd/tree/60098e776b244d2c1ec7e6bd76a628fa4378393e/src/hrd/hrl_bs_ijcnn2023/plots/neural_activity">link</a>.

## Observations
- Connectivity analysis of the trained weights: 
    - Even though we couldn't find any direct patterns using the naked eye, we found that the difference in the values of weights before and after training is significant. This suggest that the network is learning something. The plots can be found here <a href="https://github.com/CMC-lab/hrd/tree/01535ec66d591563f56759b2782e27f24fba1543/src/hrd/hrl_bs_ijcnn2023/plots/connectivity_matrix">link</a>.
    
- Visual analysis of the agent's Behavior: 
    - We observed a binary behavior where the agent moves around and eats when it's hungry else it doesn't move at all.
    - When the agent is very hungry with internal state (-0.7, -0.7), it doesn't move at all because moving also costs energy.
    - Also when the agent is completely full with internal state (0.7, 0.7), it doesn't move at all.
    - The agent starts to move when the internal state is (0.0, 0.0) and it moves towards the resource.
    - The speed of the agent seems to be affected by the internal state as well. The agent moves faster when the internal state is (0.0, 0.0) and it moves slower when the internal state is (-0.5, -0.5).

- Neural activity analysis:
    - We observed that the magnitude of the neural activity is different for different internal states. The neural activity is higher when the internal state is (0.0, 0.0) and lower when the internal state is (-0.5, -0.5). This suggest that the network is learning something and the neural activity is affected by the internal state.

## Next steps
- Running experiments with certain changes to the environment and visualize the behavior of the agent.
- Measure the neural activity of the agent and compare that with the neural activity of the brain.