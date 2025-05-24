import matplotlib.pyplot as plt
import numpy as np
import os # Added for makedirs

def _plot_sp_on_ax(ax, sp_history):
    """Helper to plot SP history on a given Matplotlib Axes object."""
    if not sp_history:
        ax.text(0.5, 0.5, "SP history is empty.", ha='center', va='center')
        ax.set_title("Survival Points (SP) Over Cycles")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Survival Points (SP)")
        return

    cycles, sp_values = zip(*sp_history)
    ax.plot(cycles, sp_values, marker='.', linestyle='-')
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

    ax.plot(loss_history, marker='.', linestyle='-')
    ax.set_title(f"{loss_name} Over Training Cycles/Epochs")
    ax.set_xlabel("Training Step/Epoch")
    ax.set_ylabel(loss_name)
    ax.grid(True)

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

def plot_full_experiment_visualization(
    sp_history, 
    phase1_loss_log, 
    phase2_loss_log, 
    probe_gb_log, 
    l1_category_map, 
    save_path=None, 
    show_plot=False
):
    """
    Plots SP history, Phase 1/2 losses, and G-B evolution for probes in a single figure.
    """
    num_probes = len(probe_gb_log.keys()) if probe_gb_log else 0
    # We expect 3 summary plots (SP, P1 Loss, P2 Loss) and 3 probe plots
    # Asserting this to match the 3x2 layout assumption
    if num_probes != 3:
        print(f"Warning: Expected 3 probe stimuli for 3x2 layout, found {num_probes}. Plot may be misaligned or incomplete.")
        # Fallback or error could be implemented here, for now, it will proceed and might error or look odd.

    # Define a 3x2 grid for the plots
    # Column 0: SP, P1 Loss, P2 Loss
    # Column 1: Probe 1, Probe 2, Probe 3 G-B evolution
    num_rows = 3
    num_cols = 2
    
    # Adjust figsize for a 3x2 layout
    subplot_width = 10  # inches
    subplot_height = 5 # inches
    fig_width = num_cols * subplot_width
    fig_height = num_rows * subplot_height

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)
    # axs is now a 2D array, e.g., axs[row, col]

    # Plot SP History in axs[0, 0]
    _plot_sp_on_ax(axs[0, 0], sp_history)

    # Plot Phase 1 Loss in axs[1, 0]
    _plot_loss_on_ax(axs[1, 0], phase1_loss_log, "Phase 1 L1 Loss")

    # Plot Phase 2 Loss in axs[2, 0]
    _plot_loss_on_ax(axs[2, 0], phase2_loss_log, "Phase 2 Total Loss")

    # Plot Probe G-B Evolution in the second column (axs[row, 1])
    if probe_gb_log and num_probes > 0:
        probe_names = list(probe_gb_log.keys()) 
        # Define styles for categories (up to 3 for now, can be extended)
        category_styles = {
            0: {'name': 'circle', 'g_color': 'blue', 'b_color': 'deepskyblue', 'marker': 'o'}, # Assuming circle is cat 0
            1: {'name': 'square', 'g_color': 'green', 'b_color': 'limegreen', 'marker': 's'}, # Assuming square is cat 1
            2: {'name': 'noise',  'g_color': 'red',   'b_color': 'salmon',      'marker': '^'}  # Assuming noise is cat 2
        }
        # Create a reverse map from name to style details if l1_category_map is used directly
        cat_name_to_style = {style_details['name']: style_details 
                               for _, style_details in category_styles.items() 
                               if style_details['name'] in l1_category_map}
        
        # Fallback for categories not in predefined styles (e.g. if l1_category_map changes)
        default_colors = plt.cm.get_cmap('tab10', len(l1_category_map)*2) # times 2 for G and B
        color_idx = 0

        for i, probe_name in enumerate(probe_names):
            if i >= num_rows: 
                break
            
            ax = axs[i, 1] 
            ax.set_title(f"G-B for Input: {probe_name.replace('probe_', '').capitalize()}")
            ax.set_xlabel("Training Cycle")
            ax.set_ylabel("G/B Value")
            ax.grid(True)
            ax.set_ylim(-0.1, 1.1)

            if not probe_gb_log[probe_name]:
                ax.text(0.5, 0.5, "No data for this probe.", ha='center', va='center')
                continue

            legend_handles = []
            # Iterate based on l1_category_map to maintain order and ensure all L1 cats are considered
            sorted_category_items = sorted(l1_category_map.items(), key=lambda item: item[1])

            for cat_name, cat_idx in sorted_category_items:
                if cat_name not in probe_gb_log[probe_name]:
                    print(f"Warning: Monitored category '{cat_name}' not found in log for probe '{probe_name}'. Skipping.")
                    continue
                
                style = cat_name_to_style.get(cat_name)

                g_data = probe_gb_log[probe_name][cat_name].get('g', [])
                b_data = probe_gb_log[probe_name][cat_name].get('b', [])

                if g_data:
                    cycles_g, values_g = zip(*g_data)
                    g_color = style['g_color'] if style else default_colors(color_idx)
                    marker = style['marker'] if style else '.'
                    line, = ax.plot(cycles_g, values_g, marker=marker, linestyle='-', color=g_color, label=f"{cat_name} (G)")
                    legend_handles.append(line)
                    if not style: color_idx +=1
                if b_data:
                    cycles_b, values_b = zip(*b_data)
                    b_color = style['b_color'] if style else default_colors(color_idx)
                    marker = style['marker'] if style else 'x'
                    line, = ax.plot(cycles_b, values_b, marker=marker, linestyle='--', color=b_color, label=f"{cat_name} (B)")
                    legend_handles.append(line)
                    if not style: color_idx +=1
            
            if legend_handles:
                ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    
    # If there are fewer than 3 probes, hide unused axes in the second column
    for i in range(num_probes, num_rows):
        if i < axs.shape[0]: # Check if the row index is valid
             axs[i, 1].axis('off')

    fig.tight_layout(rect=[0, 0, 0.93, 1]) # Adjust rect to ensure legends fit

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"Full experiment visualization saved to {save_path}")

    if show_plot:
        plt.show()
    elif not save_path:
        print("Full visualization plot generated but neither saved nor shown.")
    
    plt.close(fig)

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