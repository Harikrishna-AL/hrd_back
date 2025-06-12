import numpy as np
import matplotlib.pyplot as plt

#load the .npz file
def load_npz_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data

if __name__ == "__main__":
    # Example usage
    file_path = "hrl_bs_ijcnn2023/plots/neural_activity/neural_activations.npz"
    data = load_npz_file(file_path)
    
    layer1_act = data['activations_layer1_act'][2]
    layer1_crt = data['activations_layer1_crt'][2]
    layer2_act = data['activations_layer2_act'][2]
    layer2_crt = data['activations_layer2_crt'][2]
    output_act = data['activations_output_act'][2]
    output_crt = data['activations_output_crt'][2]
    velocity = data['velocities'][2]
    
    plot_data = {
    'layer1_act_reg' : [layer1_act[500:1000, :], layer1_act[2000:2500, :]],
    'layer1_crt_reg' : [layer1_crt[500:1000, :], layer1_crt[2000:2500, :]],
    'layer2_act_reg' : [layer2_act[500:1000, :], layer2_act[2000:2500, :]],
    'layer2_crt_reg' : [layer2_crt[500:1000, :], layer2_crt[2000:2500, :]],
    'output_act_reg' : [output_act[500:1000, :], output_act[2000:2500, :]],
    'output_crt_reg' : [output_crt[500:1000, :], output_crt[2000:2500, :]],
    'velocity_reg' : [velocity[500:1000], velocity[2000:2500]]
    }

    
    # plot nut level 0 within 500-1000 and 2000-2500 time steps 
    fig, axs = plt.subplots(7, 2, figsize=(10, 28))
    for i, (key, value) in enumerate(plot_data.items()):
        print(f"Plotting {key} with shape {value[0].shape} and {value[1].shape}")
        if i in [0,1,2,3,4]:
            axs[i, 0].imshow(value[0].T, aspect='auto', cmap='viridis', interpolation='nearest')
            axs[i, 0].set_title(f'{key} - Region 1')
            
            axs[i, 1].imshow(value[1].T, aspect='auto', cmap='viridis', interpolation='nearest')
            axs[i, 1].set_title(f'{key} - Region 2')
        
        if i in [5, 6]:
            axs[i, 0].plot(value[0], c='red', alpha=0.5, label='Critic Output Layer Activation')
            axs[i, 0].set_title(f'{key} - Region 1')
            axs[i, 0].set_xlabel('Time Steps')
            axs[i, 0].set_ylabel('Activation')
            
            axs[i, 1].plot(value[1], c='blue', alpha=0.5, label='Velocity')
            axs[i, 1].set_title(f'{key} - Region 2')
            axs[i, 1].set_xlabel('Time Steps')
            axs[i, 1].set_ylabel('Activation')

    import time
    plt.tight_layout()	
    plt.savefig("hrl_bs_ijcnn2023/plots/neural_activity/neural_activations_regions" + str(time.time()) + ".png", dpi=300)
    plt.close()
    
