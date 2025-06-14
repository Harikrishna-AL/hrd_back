# State of Analysis (SoA)

## Experiments & Analyses


### 1. **Plot the Neural Activity for all the Layers**  
**Goal:**  
To examine how the agent's neural activity evolves over time across different layers and determine if there's a correspondence between neural activation patterns and agent behavior (e.g., velocity changes).

**Details:**  
Neural activity across all layers was plotted over the time course of the agent’s trajectory. The plots clearly show fluctuations in neuron activations in relation to the agent’s movement and interactions with the environment.

**Outcome:**  
A clear correlation was observed between the agent’s velocity and neural activity across layers. Changes in agent speed both increases and decreases corresponded to noticeable shifts in neuron activations.

**Conclusion & Next Step:**  
The relationship between neural activations and velocity provides a solid foundation for further behavior-based interpretation of the network. Next, we plan to zoom in on specific regions where significant behavioral events (like stopping or sudden movement) occur to better understand how these are encoded in the network.

---

### 2. **Analyze Specific Regions of Interest (ROIs) in the Neural Activity Plots (During Stopping and Velocity Initiation)**  
**Goal:**  
To inspect regions where notable behavioral transitions occur and determine their neural signatures.

**Details:**  
Two regions were manually selected in the neural activity plots (see [Figure 1](https://github.com/CMC-lab/hrd/blob/master/src/hrd/hrl_bs_ijcnn2023/plots/neural_activity/neural_activations_regions1749791844.4416938.png)):
- **Region 1**: A velocity drop (agent slows down).
- **Region 2**: A velocity spike (agent accelerates).

Layer-wise examination of neural activity revealed that both regions showed clear changes in activation patterns. In particular, Layers 1 and 2 displayed noticeable shifts corresponding to the velocity changes.

**Outcome:**  
These ROIs confirm that the network's internal state is dynamically modulated in response to the agent's motion state. The activations in critic-output layer are evidently inversely proportional to the velocity of the agent.

**Conclusion & Next Step:**  
These observations validate that neural representations are tightly linked to motor control or velocity modulation. Moving forward, we aim to identify if these neural markers can be generalized across different trajectories or conditions.

---

### 3. **Plot the Trajectory for all the Nutrient Levels along with the Mean Positions of Both Nutrients**  
**Goal:**  
To understand the agent’s spatial exploration pattern and its relation to resource locations over time.

**Details:**  
The agent's full trajectory was visualized alongside the mean positions of both red and blue nutrients ([Figure 2](https://github.com/CMC-lab/hrd/blob/master/src/hrd/hrl_bs_ijcnn2023/plots/neural_activity/positions_trajectory.png)).  
- The **green dot** denotes the agent's starting point.  
- The **black cross** marks the endpoint.  
- The **red and blue dots** indicate the average nutrient locations.

**Outcome:**  
In all trajectories, the agent tends to follow a spiral-like path in its exploration strategy. The nutrient locations are well-distributed, and their average positions are consistently marked. Interestingly, in extreme scenarios, the agent's exploratory behavior appears less pronounced.

**Conclusion & Next Step:**  
The spiral trajectory indicates a structured exploration strategy. Need to discuss more on extreme scenarios. A more quantitative metric for exploration could be computed to validate this further.