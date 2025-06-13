import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto


class FoodClass(Enum):
    BLUE = 1
    RED = 2

def load_npz_file(file_path):
    """
    Load a .npz file and return its contents.
    
    Parameters:
    file_path (str): Path to the .npz file.
    
    Returns:
    dict: Dictionary containing the loaded data.
    """
    data = np.load(file_path, allow_pickle=True)
    return data

if __name__ == "__main__":
    # Example usage
    file_path = "hrl_bs_ijcnn2023/plots/neural_activity/neural_activations.npz"
    data = load_npz_file(file_path)
    
    print(data['positions'].shape)
    
    positions = data['positions']
    objects = data['objects']
    
    x = np.array(objects[:, :, :, 0], dtype=np.float32)
    y = np.array(objects[:, :, :, 1], dtype=np.float32)
    
    color_map = {
        'FoodClass.BLUE': 1,
        'FoodClass.RED': 2
    }
    # Extract the color values (should be integers: 1 for BLUE, 2 for RED)
    color = np.vectorize(lambda c: color_map[str(c)])(objects[:, :, :, 2])

    blue_mask = color == 1
    red_mask = color == 2
    print(blue_mask[0,0])
    print("BLUE mask shape:", blue_mask.shape)
    print("RED mask shape:", red_mask.shape)

    # Get indices of BLUE and RED objects where the mask is True
    x_blue = np.where(blue_mask, x, np.nan)
    y_blue = np.where(blue_mask, y, np.nan)
    x_red = np.where(red_mask, x, np.nan)
    y_red = np.where(red_mask, y, np.nan)

    # Compute average x and y ignoring nan
    blue_avg = np.nanmean(np.stack([x_blue, y_blue], axis=-1), axis=2)  # shape: (5, 6000, 2)
    red_avg  = np.nanmean(np.stack([x_red, y_red], axis=-1), axis=2)    # shape: (5, 6000, 2)

    print("BLUE average shape:", blue_avg.shape)  # (5, 6000, 2)
    print("RED average shape:", red_avg.shape)
            
    fig, ax = plt.subplots(5,1,figsize=(5, 20))
    for i in range(5):
        # plot the trajectory for all the nutrient values (5)
        #for each nutrient postions are as 6000, 2. plot 2 x,y position for each 6000 time steps and track the trajectory
        # make 2d plot of trajectory for each nutrient
        ax[i].plot(positions[i, :, 0], positions[i, :, 1], label=f'Nutrient {i+1}')
        # ax[i].plot(positions[i], label=f'Nutrient {i+1}')
        ax[i].set_title(f'Nutrient {i+1} Trajectory')
        ax[i].set_xlabel('X axis Position')
        ax[i].set_ylabel('Y axis Position')
        
        ax[i].scatter(blue_avg[i, :, 0], blue_avg[i, :, 1], c='blue', label='Blue Objects', s=3)
        ax[i].scatter(red_avg[i, :, 0], red_avg[i, :, 1], c='red', label='Red Objects', s=3)
        ax[i].legend()
    plt.tight_layout()
    save_path = "hrl_bs_ijcnn2023/plots/neural_activity/positions_trajectory.png"
    plt.savefig(save_path)
    