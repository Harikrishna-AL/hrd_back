# State of Analysis (SoA)

##  Project Goal  
To analyze the *homeostatic behavior* and *neural activity* of an artificial agent in a two-resource environment, and compare this with neural patterns in biological systems. The agent must maintain internal resource levels (red and blue) through foraging behavior, guided by interoceptive feedback and reinforcement learning. This project focuses on understanding:
- How the agent learns homeostatic control
- How its behavior adapts to internal states
- How its neural activity reflects these changes

---

##  Experiments & Analyses  

### 1. **Characterizing Agent & Environment**
**Goal:** Understand the structure and sensory input composition of the agent.

**Details:**
- **Inputs:**
  - Exteroception: 40-D (external sensory input)
  - Proprioception: 27-D (body movement/position)
  - Interoception: 2-D (internal state: [blue, red])
- **Full documentation:** [network characterization notes](https://github.com/CMC-lab/hrd/blob/01535ec66d591563f56759b2782e27f24fba1543/notes/network_characterization.md)

**Outcome:** Established a foundational understanding of agent perception and input dimensionality.

**Conclusion & Next Step:**  
We can now begin interpreting behavior and neural activity in terms of specific inputs.

---

### 2. **Connectivity Analysis of Trained Weights**
**Goal:** Investigate whether training leads to structural changes in the network’s weight connectivity.

**Method:**
- Visualized weight matrices as heatmaps.
- Compared pre- and post-training values.

**Outcome:**
- No immediately visible patterns.
- However, significant changes in weight magnitudes post-training.
- **Plots available:** [connectivity matrices](https://github.com/CMC-lab/hrd/tree/01535ec66d591563f56759b2782e27f24fba1543/src/hrd/hrl_bs_ijcnn2023/plots/connectivity_matrix)

- **Code Used:** [connectivity analysis code](https://github.com/CMC-lab/hrd/blob/111b0a5a900452ceaa2a43a8d2f714e29620bc0e/src/hrd/hrl_bs_ijcnn2023/connectivity.ipynb)

**Conclusion & Next Step:**  
Training does cause structural changes, implying learning. We should next quantify these differences or use clustering to extract structured patterns if they exist.

---

### 3. **Visual Behavior Analysis**
**Goal:** Examine how agent behavior changes with varying initial internal states.

**Method:**
- Ran rollouts from different internal state initializations.
- Observed movement and foraging behavior.

**Outcome:**
- Binary behavior emerged:
  - Very low energy → no movement (e.g., state = (-0.7, -0.7))
  - Very high energy → no movement (e.g., state = (0.7, 0.7))
  - Mid-range (e.g., (0.0, 0.0)) → active foraging and faster movement
- Movement speed correlated with internal state level

**Conclusion & Next Step:**  
Behavior is strongly modulated by internal state. Suggests homeostatic control is encoded. We will now link this behavior to neural activity and policy decision-making more directly.

---

### 4. **Neural Activity Analysis**
**Goal:** Determine if internal state affects neural activation patterns.

**Method:**
- Recorded activations from actor and critic networks.
- Reduced to 30 dimensions using PCA.
- Plotted as heatmaps under different internal states.

**Outcome:**
- Clear variation in neural activity magnitude across states.
  - Mid-range internal state → highest activity
  - Lower energy states → suppressed activity
- **Visualizations:** [neural activity plots](https://github.com/CMC-lab/hrd/tree/60098e776b244d2c1ec7e6bd76a628fa4378393e/src/hrd/hrl_bs_ijcnn2023/plots/neural_activity)
- **Code Used:** [neural activity analysis code](https://github.com/CMC-lab/hrd/blob/111b0a5a900452ceaa2a43a8d2f714e29620bc0e/src/hrd/test_env.py)

**Conclusion & Next Step:**  
Neural activation strength is modulated by internal state, mirroring biological motivational systems. Next, we plan to analyze temporal dynamics of activity and compare this to real neural recordings from biological organisms.

---

## Key Takeaways & Next Steps

### Key Conclusions:
- Agent develops **adaptive behaviors** tied to its internal state, demonstrating homeostatic control.
- **Neural activity** varies systematically with internal state, suggesting functional encoding.
- **Weight changes** post-training confirm that learning occurs, though more analysis is needed to interpret structure.

### Next Steps:
- Introduce controlled perturbations to the environment (e.g., changing reward delays or cost of movement).
- Compare artificial neural dynamics with real brain recordings.
- Investigate causality by lesioning parts of the network or input modalities.
