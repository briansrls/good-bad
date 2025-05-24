import matplotlib.pyplot as plt
import numpy as np
import os # Added for makedirs
import torch

def _plot_sp_on_ax(ax, sp_history):
    """Helper to plot SP history on a given Matplotlib Axes object."""
    if not sp_history:
        ax.text(0.5, 0.5, "SP history is empty.", ha='center', va='center')
        ax.set_title("Survival Points (SP) Over Cycles")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Survival Points (SP)")
        return

    cycles, sp_values = zip(*sp_history)
    ax.plot(cycles, sp_values, marker='.', linestyle='-', color='black')
    ax.set_title("Survival Points (SP) Over Cycles")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Survival Points (SP)")
    ax.grid(True)
    ax.set_ylim(bottom=0)

def _plot_loss_on_ax(ax, loss_history, loss_name="Total Loss"):
    """Helper to plot loss history on a given Matplotlib Axes object."""
    if not loss_history:
        ax.text(0.5, 0.5, f"{loss_name} history is empty.", ha='center', va='center')
        ax.set_title(f"{loss_name} Over Training Cycles/Epochs")
        ax.set_xlabel("Training Step/Epoch")
        ax.set_ylabel(loss_name)
        return

    ax.plot(loss_history, marker='.', linestyle='-', color='purple')
    ax.set_title(f"{loss_name} Over Training Cycles/Epochs")
    ax.set_xlabel("Training Step/Epoch")
    ax.set_ylabel(loss_name)
    ax.grid(True)

def _plot_conv_weights_on_ax(ax, weights_tensor, num_filters_to_show=16, title="Conv1 Weights"):
    """Helper to plot convolutional filter weights on a given Matplotlib Axes object."""
    ax.set_title(title)
    ax.axis('off') # Turn off axis numbers and ticks for image display

    if weights_tensor is None or not isinstance(weights_tensor, np.ndarray) and not torch.is_tensor(weights_tensor):
        ax.text(0.5, 0.5, "Weights not available or invalid format.", ha='center', va='center')
        return

    if torch.is_tensor(weights_tensor):
        weights_tensor = weights_tensor.cpu().numpy()
    
    # Assuming weights_tensor shape: (out_channels, in_channels, kernel_h, kernel_w)
    # For L1Network.conv1, in_channels is 1 (grayscale)
    if weights_tensor.ndim != 4 or weights_tensor.shape[1] != 1:
        ax.text(0.5, 0.5, f"Unexpected weights shape: {weights_tensor.shape}. Expected (N,1,H,W).", ha='center', va='center')
        return

    num_filters = weights_tensor.shape[0]
    filters_to_show = min(num_filters, num_filters_to_show)
    
    # Determine grid size for displaying filters
    cols = int(np.ceil(np.sqrt(filters_to_show)))
    rows = int(np.ceil(filters_to_show / cols))
    
    # Create a larger image to hold all filter images in a grid
    kernel_h, kernel_w = weights_tensor.shape[2], weights_tensor.shape[3]
    padding = 1 # pixels between filters
    grid_img_h = rows * kernel_h + (rows - 1) * padding
    grid_img_w = cols * kernel_w + (cols - 1) * padding
    grid_image = np.zeros((grid_img_h, grid_img_w))

    for i in range(filters_to_show):
        row_idx = i // cols
        col_idx = i % cols
        
        filter_img = weights_tensor[i, 0, :, :] # Get the i-th filter, first (and only) input channel
        
        # Normalize filter to [0, 1] for visualization
        f_min, f_max = filter_img.min(), filter_img.max()
        if f_max > f_min: # Avoid division by zero if filter is flat
            filter_img_normalized = (filter_img - f_min) / (f_max - f_min)
        else:
            filter_img_normalized = np.zeros_like(filter_img) # Or np.ones_like * 0.5 for gray
            
        # Calculate start and end positions in the grid image
        r_start = row_idx * (kernel_h + padding)
        r_end = r_start + kernel_h
        c_start = col_idx * (kernel_w + padding)
        c_end = c_start + kernel_w
        
        grid_image[r_start:r_end, c_start:c_end] = filter_img_normalized
        
    if filters_to_show > 0:
        ax.imshow(grid_image, cmap='gray', interpolation='nearest')
    else:
        ax.text(0.5,0.5, "No filters to show.", ha='center', va='center')

def _plot_stimulus_sample_on_ax(ax, stimulus_batch_sample, num_samples_to_show=4, title="Current Stimuli Batch"):
    """Helper to plot a sample of stimuli on a given Matplotlib Axes object."""
    ax.set_title(title)
    ax.axis('off')

    if stimulus_batch_sample is None or not torch.is_tensor(stimulus_batch_sample) or stimulus_batch_sample.ndim != 4:
        ax.text(0.5, 0.5, "Stimuli sample not available.", ha='center', va='center')
        return

    images = stimulus_batch_sample.cpu().numpy()
    # Assuming images shape: (batch_size, channels, height, width)
    # And channels is 1 for grayscale
    if images.shape[1] != 1:
        ax.text(0.5, 0.5, "Stimuli sample has unexpected channel count.", ha='center', va='center')
        return

    num_actual_samples = min(images.shape[0], num_samples_to_show)
    if num_actual_samples == 0:
        ax.text(0.5, 0.5, "No stimuli in sample.", ha='center', va='center')
        return

    # Determine grid size for displaying samples (e.g., 2x2 for 4 samples)
    cols = int(np.ceil(np.sqrt(num_actual_samples)))
    rows = int(np.ceil(num_actual_samples / cols))
    
    img_h, img_w = images.shape[2], images.shape[3]
    padding = 2
    grid_img_h = rows * img_h + (rows - 1) * padding
    grid_img_w = cols * img_w + (cols - 1) * padding
    grid_image = np.zeros((grid_img_h, grid_img_w))

    for i in range(num_actual_samples):
        row_idx = i // cols
        col_idx = i % cols
        img = images[i, 0, :, :] # Get i-th image, first channel
        
        r_start = row_idx * (img_h + padding)
        r_end = r_start + img_h
        c_start = col_idx * (img_w + padding)
        c_end = c_start + img_w
        grid_image[r_start:r_end, c_start:c_end] = img
        
    ax.imshow(grid_image, cmap='gray', interpolation='nearest')

def _plot_weights_histogram_on_ax(ax, weights_tensor, title="Weights Histogram"):
    """Helper to plot a histogram of weights on a given Matplotlib Axes object."""
    ax.set_title(title)
    if weights_tensor is None or not isinstance(weights_tensor, np.ndarray) and not torch.is_tensor(weights_tensor):
        ax.text(0.5, 0.5, "Weights not available.", ha='center', va='center'); return
    
    if torch.is_tensor(weights_tensor):
        weights_data = weights_tensor.detach().cpu().numpy().flatten()
    else:
        weights_data = weights_tensor.flatten()
        
    ax.hist(weights_data, bins=50, color='orange', alpha=0.75)
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)

def _plot_current_batch_gb_avg_on_ax(ax, current_l1_gb_outputs_batch_avg, l1_category_map, title="L1 G-B Avg for Current Batch"):
    ax.set_title(title)
    if current_l1_gb_outputs_batch_avg is None or not torch.is_tensor(current_l1_gb_outputs_batch_avg):
        ax.text(0.5, 0.5, "Batch G-B avg not available.", ha='center', va='center'); return

    gb_values = current_l1_gb_outputs_batch_avg.cpu().numpy() # Shape: (num_categories, 2)
    num_categories = gb_values.shape[0]

    if isinstance(l1_category_map, dict):
        cat_names = ["Unknown"] * num_categories
        for name, idx in l1_category_map.items():
            if 0 <= idx < num_categories:
                 # If multiple names map to an index (e.g. line/circle), take the first one for simplicity here
                if cat_names[idx] == "Unknown" or 'circle' in name.lower(): # Prioritize primary names
                    cat_names[idx] = name.replace("probe_","").capitalize()
    else:
        cat_names = [f"Cat {i}" for i in range(num_categories)]

    g_vals = gb_values[:, 0]
    b_vals = gb_values[:, 1]
    x = np.arange(num_categories)
    width = 0.35

    rects1 = ax.bar(x - width/2, g_vals, width, label='Avg G', color='lightgreen')
    rects2 = ax.bar(x + width/2, b_vals, width, label='Avg B', color='lightcoral')
    ax.set_ylabel('Avg G/B Value')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=45, ha="right")
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def plot_experiment_summary(sp_history, phase1_loss_log, phase2_loss_log, save_path=None, show_plot=False):
    """
    Plots SP history, Phase 1 loss, and Phase 2 loss in a single figure with subplots.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 18)) # 3 rows, 1 column

    # Plot SP History
    _plot_sp_on_ax(axs[0], sp_history)

    # Plot Phase 1 Loss
    _plot_loss_on_ax(axs[1], phase1_loss_log, "Phase 1 L1 Loss")

    # Plot Phase 2 Loss
    _plot_loss_on_ax(axs[2], phase2_loss_log, "Phase 2 Total Loss")

    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"Experiment summary plot saved to {save_path}")

    if show_plot:
        plt.show()
    elif not save_path:
        print("Plot generated but neither saved nor shown.")
    
    plt.close(fig) # Close the figure

def plot_sp_history(sp_history, save_path=None, show_plot=False):
    """
    Plots the Survival Points (SP) over cycles.
    (This function can still be used for individual SP plots if needed)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_sp_on_ax(ax, sp_history)

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name: 
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"SP history plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    elif not save_path: 
        print("Plot generated but neither saved nor shown.")

    plt.close(fig)

def plot_loss_history(loss_history, loss_name="Total Loss", save_path=None, show_plot=False):
    """
    Plots a generic loss history over cycles/epochs.
    (This function can still be used for individual loss plots if needed)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_loss_on_ax(ax, loss_history, loss_name)

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"{loss_name} history plot saved to {save_path}")

    if show_plot:
        plt.show()
    elif not save_path:
        print("Plot generated but neither saved nor shown.")
        
    plt.close(fig)

def plot_gb_values_example(gb_values_tensor, category_map, title="G-B Values Example", save_path=None, show_plot=False):
    """
    Plots G and B values for categories from a gb_values_tensor (batch_size, num_categories, 2).
    This is a simplified example showing the G/B for the *first item in a batch*.

    Args:
        gb_values_tensor (torch.Tensor or np.ndarray): Shape (batch_size, num_categories, 2) or (num_categories,2).
        category_map (dict): Maps category names to indices, e.g., {'circle': 0, 'noise': 1}.
                               Or a list of category names if indices match list position.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. 
        show_plot (bool): Whether to display the plot.
    """
    if hasattr(gb_values_tensor, 'detach'): # Check if it's a PyTorch tensor
        gb_values = gb_values_tensor.detach().cpu().numpy()
    else:
        gb_values = np.array(gb_values)

    if gb_values.ndim == 3: # batch, num_cat, 2
        if gb_values.shape[0] == 0:
            print("G-B values tensor is empty (batch size 0). Cannot plot.")
            return
        gb_values_item = gb_values[0] # Take first item in batch
    elif gb_values.ndim == 2: # num_cat, 2
        gb_values_item = gb_values
    else:
        raise ValueError("gb_values_tensor must have shape (batch, num_cat, 2) or (num_cat, 2)")

    num_categories = gb_values_item.shape[0]
    if isinstance(category_map, dict):
        cat_names = [None] * num_categories
        for name, idx in category_map.items():
            if 0 <= idx < num_categories:
                cat_names[idx] = name
        cat_names = [name if name is not None else f"Cat_{i}" for i, name in enumerate(cat_names)]
    elif isinstance(category_map, list) and len(category_map) == num_categories:
        cat_names = category_map
    else:
        cat_names = [f"Category {i}" for i in range(num_categories)]

    g_vals = gb_values_item[:, 0]
    b_vals = gb_values_item[:, 1]

    x = np.arange(num_categories)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, g_vals, width, label='Goodness (G)', color='skyblue')
    rects2 = ax.bar(x + width/2, b_vals, width, label='Badness (B)', color='salmon')

    ax.set_ylabel('Activation Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1) # G/B values are between 0 and 1

    fig.tight_layout()
    if save_path:
        # Ensure directory exists
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"G-B values plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    elif not save_path:
         print("Plot generated but neither saved nor shown.")

    plt.close() # Close the figure

def plot_probe_gb_evolution(probe_gb_log, l1_category_map, save_path=None, show_plot=False):
    """
    Plots the evolution of G-B values for probe stimuli over training cycles.

    Args:
        probe_gb_log (dict): Logged G-B values. 
                             Format: {probe_name: {cat_name: {'g': [(cycle, val)], 'b': [(cycle, val)]}}}
        l1_category_map (dict): Maps L1 category names to indices.
        save_path (str, optional): Path to save the combined plot.
        show_plot (bool): Whether to display the plot.
    """
    if not probe_gb_log:
        print("Probe G-B log is empty. Cannot plot.")
        return

    probe_names = list(probe_gb_log.keys())
    num_probes = len(probe_names)
    if num_probes == 0:
        print("No probes found in log. Cannot plot.")
        return

    category_names = list(l1_category_map.keys()) # Order of categories for plotting lines

    fig, axs = plt.subplots(num_probes, 1, figsize=(12, 6 * num_probes), squeeze=False)
    axs = axs.flatten() # Ensure axs is always an array

    for i, probe_name in enumerate(probe_names):
        ax = axs[i]
        ax.set_title(f"G-B Evolution for Input: {probe_name.replace('probe_', '').capitalize()}")
        ax.set_xlabel("Training Cycle")
        ax.set_ylabel("G/B Value")
        ax.grid(True)
        ax.set_ylim(-0.1, 1.1) # G/B values are typically 0-1

        if not probe_gb_log[probe_name]:
            ax.text(0.5, 0.5, "No data for this probe.", ha='center', va='center')
            continue

        legend_handles = []

        for cat_name in category_names:
            if cat_name not in probe_gb_log[probe_name]:
                print(f"Warning: Category '{cat_name}' not found in log for probe '{probe_name}'. Skipping.")
                continue
            
            g_data = probe_gb_log[probe_name][cat_name].get('g', [])
            b_data = probe_gb_log[probe_name][cat_name].get('b', [])

            if g_data:
                cycles_g, values_g = zip(*g_data)
                line, = ax.plot(cycles_g, values_g, marker='.', linestyle='-', label=f"{cat_name} (G)")
                legend_handles.append(line)
            if b_data:
                cycles_b, values_b = zip(*b_data)
                line, = ax.plot(cycles_b, values_b, marker='x', linestyle='--', label=f"{cat_name} (B)")
                legend_handles.append(line)
        
        if legend_handles:
            ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"Probe G-B evolution plot saved to {save_path}")

    if show_plot:
        plt.show()
    elif not save_path:
        print("Probe G-B plot generated but neither saved nor shown.")
    
    plt.close(fig)

def _plot_gb_evolution_single_probe_on_ax(ax, probe_gb_log, probe_name, l1_category_map_local):
    ax.set_title(f"G-B for Input: {probe_name.replace('probe_', '').capitalize()}")
    ax.set_xlabel("Training Cycle"); ax.set_ylabel("G/B Value"); ax.grid(True); ax.set_ylim(-0.1, 1.1)
    category_styles_local = {0:{'name':'circle','g_color':'blue','b_color':'deepskyblue','marker':'o'},1:{'name':'square','g_color':'green','b_color':'limegreen','marker':'s'},2:{'name':'noise','g_color':'red','b_color':'salmon','marker':'^'}}
    cat_name_to_style_local = {sd['name']:sd for _,sd in category_styles_local.items() if sd['name'] in l1_category_map_local}
    default_colors_local = plt.cm.get_cmap('tab10',len(l1_category_map_local)*2); color_idx_probe_local=0; legend_handles_local=[]
    if not probe_gb_log.get(probe_name) or not any(probe_gb_log[probe_name].get(cn,{}).get('g',[]) or probe_gb_log[probe_name].get(cn,{}).get('b',[]) for cn in probe_gb_log[probe_name]):
        ax.text(0.5,0.5,"No data yet.",ha='center',va='center'); return
    sorted_cat_items_local = sorted(l1_category_map_local.items(),key=lambda item:item[1])
    for cat_n,cat_i in sorted_cat_items_local:
        if cat_n not in probe_gb_log[probe_name]: continue
        sty=cat_name_to_style_local.get(cat_n); gd=probe_gb_log[probe_name][cat_n].get('g',[]); bd=probe_gb_log[probe_name][cat_n].get('b',[])
        if gd: 
            cg,vg=zip(*gd);gc=sty['g_color'] if sty else default_colors_local(color_idx_probe_local);mk=sty['marker'] if sty else '.'
            l, =ax.plot(cg,vg,marker=mk,ls='-',color=gc,label=f"{cat_n}(G)"); legend_handles_local.append(l);
            color_idx_probe_local+=0 if sty else 1
        if bd: 
            cb,vb=zip(*bd);bc=sty['b_color'] if sty else default_colors_local(color_idx_probe_local);mk=sty['marker'] if sty else 'x'
            l, =ax.plot(cb,vb,marker=mk,ls='--',color=bc,label=f"{cat_n}(B)"); legend_handles_local.append(l);
            color_idx_probe_local+=0 if sty else 1
    if legend_handles_local: ax.legend(handles=legend_handles_local,loc='center left',bbox_to_anchor=(1.01,0.5),borderaxespad=0.)

def plot_full_experiment_visualization(
    fig, 
    axs, # Expected to be 4x3 now
    sp_history, 
    phase1_loss_log, 
    phase2_loss_log, 
    probe_gb_log, 
    l1_category_map, 
    l1_conv1_weights=None,
    l1_conv2_weights=None,      
    current_stimuli_sample=None, 
    current_l1_gb_outputs_batch_avg=None, # New arg
    save_path_final=None,
):
    """
    Updates 4x3 grid.
    Row 0: SP, Probe Circle G-B, L1 Conv1 Filters
    Row 1: P1 Loss, Probe Square G-B, Current Batch Stimuli Sample
    Row 2: P2 Loss, Probe Noise G-B, L1 Conv2 Weights Hist.
    Row 3: L1 G-B Output for Current Batch (Col 0), empty (Col 1), empty (Col 2)
    """
    # Clear all axes
    for r_idx in range(axs.shape[0]): # 4 rows
        for c_idx in range(axs.shape[1]): # 3 cols
            axs[r_idx, c_idx].clear()

    # Row 0
    _plot_sp_on_ax(axs[0, 0], sp_history)
    # Probe Circle G-B (assuming 'probe_circle' is first if keys are sorted or consistently ordered)
    probe_names_list = list(probe_gb_log.keys()) # Need consistent order for probes
    if len(probe_names_list) > 0: _plot_gb_evolution_single_probe_on_ax(axs[0, 1], probe_gb_log, probe_names_list[0], l1_category_map)
    else: axs[0,1].text(0.5,0.5,"Probe data missing.", ha='center',va='center')
    num_conv1_filters = l1_conv1_weights.shape[0] if hasattr(l1_conv1_weights, 'shape') else 0
    _plot_conv_weights_on_ax(axs[0, 2], l1_conv1_weights, num_filters_to_show=min(16, num_conv1_filters), title=f"L1 Conv1 Filters (Top {min(16, num_conv1_filters)})")

    # Row 1
    _plot_loss_on_ax(axs[1, 0], phase1_loss_log, "Phase 1 L1 Loss")
    if len(probe_names_list) > 1: _plot_gb_evolution_single_probe_on_ax(axs[1, 1], probe_gb_log, probe_names_list[1], l1_category_map)
    else: axs[1,1].text(0.5,0.5,"Probe data missing.", ha='center',va='center')
    _plot_stimulus_sample_on_ax(axs[1, 2], current_stimuli_sample, num_samples_to_show=4, title="Batch Stimuli Sample")

    # Row 2
    _plot_loss_on_ax(axs[2, 0], phase2_loss_log, "Phase 2 Total Loss")
    if len(probe_names_list) > 2: _plot_gb_evolution_single_probe_on_ax(axs[2, 1], probe_gb_log, probe_names_list[2], l1_category_map)
    else: axs[2,1].text(0.5,0.5,"Probe data missing.", ha='center',va='center')
    _plot_weights_histogram_on_ax(axs[2, 2], l1_conv2_weights, title="L1 Conv2 Weights Hist.")

    # Row 3
    _plot_current_batch_gb_avg_on_ax(axs[3, 0], current_l1_gb_outputs_batch_avg, l1_category_map)
    axs[3, 1].axis('off') # Hide unused subplot
    axs[3, 2].axis('off') # Hide unused subplot
    
    fig.tight_layout(rect=[0, 0, 0.93, 0.95]) 
    fig.canvas.draw_idle()
    plt.pause(0.01)

    if save_path_final:
        dir_name = os.path.dirname(save_path_final)
        if dir_name: os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path_final)
        print(f"Full experiment visualization saved to {save_path_final}")

if __name__ == '__main__':
    print("Testing visualization functions...")

    # Test SP History Plot
    test_sp_hist = [(i, 100 - i*0.5 + np.random.randn()*5) for i in range(50)]
    test_sp_hist = [(c, max(0, sp)) for c,sp in test_sp_hist] # ensure positive
    # plot_sp_history(test_sp_hist, save_path="./sp_history_test.png")
    print("SP history plot test (call uncommented to view/save).")

    # Test Loss History Plot
    test_loss_hist = [1/(i+1) + np.random.rand()*0.1 for i in range(50)]
    # plot_loss_history(test_loss_hist, loss_name="Test L1 Loss", save_path="./loss_history_test.png")
    print("Loss history plot test (call uncommented to view/save).")

    # Test G-B Values Plot
    # Example gb_values tensor (batch_size=1, num_categories=3, 2 values G/B)
    example_gb = np.array([[[0.8, 0.1], [0.2, 0.9], [0.6, 0.5]]]) 
    cat_map = { 'Circle': 0, 'Square': 1, 'Noise': 2 }
    # plot_gb_values_example(example_gb, cat_map, title="Example G-B for Stimulus X") # save_path="./gb_values_test.png"

    example_gb_list_cats = np.array([[0.1,0.8],[0.7,0.7],[0.5,0.2],[0.9,0.1]])
    cat_list = ['Threat A', 'Opportunity B', 'Ambiguous C', 'Neutral D']
    # plot_gb_values_example(example_gb_list_cats, cat_list, title="Example G-B with List Categories")
    print("G-B values plot test (call uncommented to view/save).")

    print("Visualization tests complete. Uncomment plotting calls in if __name__ == '__main__' to generate plots.") 