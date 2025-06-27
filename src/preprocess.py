import numpy as np
import matplotlib.pyplot as plt

def get_train_names(metadata, chroma_dir):
    # value_counts = metadata['raga name'].value_counts()
    train_files=list(set(metadata['audio_file']))
    return train_files


def plot_id_vs_ood(entropy, entropy_2):
    # Define bins using min/max from both arrays
    all_values = np.concatenate([entropy, entropy_2])  
    bins = np.histogram_bin_edges(all_values, bins=50)  # Adjust bins as needed

    # Get histogram counts for each dataset
    counts_entropy, _ = np.histogram(entropy, bins=bins)
    counts_entropy_2, _ = np.histogram(entropy_2, bins=bins)

    # Find overlap (minimum of the two counts at each bin)
    overlap_counts = np.minimum(counts_entropy, counts_entropy_2)

    # Define color-blind-friendly colors
    ood_color = "#0077BB"  # Blue
    id_color = "#EE7733"   # Orange
    overlap_color = "#CC3311"  # Light Blue (for overlap)

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.bar(bins[:-1], counts_entropy, width=np.diff(bins), alpha=0.6, label="OOD", color=ood_color, align="edge")
    plt.bar(bins[:-1], counts_entropy_2, width=np.diff(bins), alpha=0.6, label="ID", color=id_color, align="edge")
    plt.bar(bins[:-1], overlap_counts, width=np.diff(bins), alpha=0.5, label="Overlap", color=overlap_color, align="edge")

    # Labels and legend
    plt.xlabel("Entropy Value")
    plt.ylabel("Count")
    plt.title("Binned Entropy Distribution")
    plt.legend()
    plt.show()
