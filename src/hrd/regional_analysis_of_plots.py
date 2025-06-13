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
    
    print(data['positions'].shape)
    
    plot_idx = 1
    
    layer1_act = data['activations_layer1_act'][plot_idx]
    layer1_crt = data['activations_layer1_crt'][plot_idx]
    layer2_act = data['activations_layer2_act'][plot_idx]
    layer2_crt = data['activations_layer2_crt'][plot_idx]
    output_act = data['activations_output_act'][plot_idx]
    output_crt = data['activations_output_crt'][plot_idx]
    velocity = data['velocities'][plot_idx]
    
    reg1_start = 3000
    reg1_end = 3000 + 300
    reg2_start = 3300
    reg2_end = 3300 + 300
    
    plot_data = {
    'layer1_act_reg' : [layer1_act[reg1_start:reg1_end, :], layer1_act[reg2_start:reg2_end, :]],
    'layer1_crt_reg' : [layer1_crt[reg1_start:reg1_end, :], layer1_crt[reg2_start:reg2_end, :]],
    'layer2_act_reg' : [layer2_act[reg1_start:reg1_end, :], layer2_act[reg2_start:reg2_end, :]],
    'layer2_crt_reg' : [layer2_crt[reg1_start:reg1_end, :], layer2_crt[reg2_start:reg2_end, :]],
    'output_act_reg' : [output_act[reg1_start:reg1_end, :], output_act[reg2_start:reg2_end, :]],
    'output_crt_reg' : [output_crt[reg1_start:reg1_end, :], output_crt[reg2_start:reg2_end, :]],
    # 'velocity_reg' : [velocity[500:1000], velocity[2000:2500]]
    'velocity_reg' : [velocity[reg1_start:reg1_end], velocity[reg2_start:reg2_end]]
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
            
            if i==6:
                axs[i, 0].set_xlim(0, reg1_end - reg1_start)
                axs[i,0].set_ylim(0, 0.8)
            else:
                axs[i, 0].set_xlim(0, reg1_end - reg1_start)
            
            axs[i, 1].plot(value[1], c='blue', alpha=0.5, label='Velocity')
            axs[i, 1].set_title(f'{key} - Region 2')
            axs[i, 1].set_xlabel('Time Steps')
            axs[i, 1].set_ylabel('Activation')
            
            if i == 6:
                axs[i, 1].set_ylim(0, 0.8)
                axs[i, 1].set_xlim(0, reg2_end - reg2_start)
            else:
                axs[i, 1].set_xlim(0, reg2_end - reg2_start)

    import time
    plt.tight_layout()	
    plt.savefig("hrl_bs_ijcnn2023/plots/neural_activity/neural_activations_regions" + str(time.time()) + ".png", dpi=300)
    plt.close()
    
