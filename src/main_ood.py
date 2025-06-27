import sys
sys.path.append("/Pim_XAI/src")
import numpy as np
import tensorflow as tf
# import model as baseline_model
from preprocess import load_required_filenames, load_metadata,create_train_val_test_splits
from data_loader import get_x_y_data
from model import make_model
from trainer import train_model, evaluate_model
from src.mc_dropout import MCDropoutModel
from src.mc_dropout import mc_dropout_predictions, batchwise_mc_dropout
from src.preprocess import get_train_names
from src.preprocess import plot_id_vs_ood

def main():
    # Path to chromagram files, metadata and list containing required Raga names
    chroma_dir = "path_to_folder_containing_chromagram_features"
    metadata_path = "path_to_metadata_csv"
    required_ragas=['Bhairavi', 'Bihag', 'Des', 'Jog', 'Kedar', 'Khamaj', 'Malkauns',
        'Maru_bihaag', 'Nayaki_kanada', 'Shuddha_kalyan', 'Sohni', 'Yaman']
    required_ragas_ood=['Bageshri', 'Bhopali','Jog_kauns', 'Mishra_khamaj', 'Puriya_kalyan']

    # Load filenames and metadata
    metadata = load_metadata(metadata_path)
    filenames_id, metadata_id = load_required_filenames(metadata, required_ragas)
    filenames_ood, metadata_ood = load_required_filenames(metadata, required_ragas_ood)

    # Split data into train, val, and test sets
    train_files=get_train_names(metadata_id, chroma_dir)
    ood_files=get_train_names(metadata_ood, chroma_dir)
    # train_files,val_files,test_files=create_train_val_test_splits(metadata, chroma_dir)

    ##load chromagrams and labels
    print("Getting ID Files!")
    x_train,y_train,class_names=get_x_y_data(train_files, chroma_dir, metadata)
    print("Getting OOD Files!")
    x_ood,y_val,_=get_x_y_data(ood_files, chroma_dir, metadata)

    num_samples = x_ood.shape[0]
    selected_indices = np.random.choice(x_train.shape[0], num_samples, replace=False)
    x_id = x_train[selected_indices]

    # Build model
    model = make_model(input_shape=x_train.shape[1:], num_classes=y_train.shape[1])
    mc_model = MCDropoutModel(model)


    n_batch = 50  # Number of samples per batch
    num_samples = 50  # Number of MC passes
    # Get MC Dropout predictions
    mc_preds_batch_ood = batchwise_mc_dropout(mc_model, x_ood, num_samples, n_batch) # Shape: (50, 20, num_classes)
    mc_preds_batch_id= batchwise_mc_dropout(mc_model, x_id, num_samples, n_batch)


    mc_preds_ood = np.concatenate(mc_preds_batch_ood, axis=1)
    mc_preds_id= np.concatenate(mc_preds_batch_id, axis=1)


    mean_probs_ood = np.mean(tf.nn.softmax(mc_preds_ood), axis=0)  # Shape (20,12)
    entropy_ood = -np.sum(mean_probs_ood * np.log(mean_probs_ood + 1e-10), axis=1)  # Shape (20,)
    mean_probs_id = np.mean(tf.nn.softmax(mc_preds_id), axis=0)  # Shape (20,12)
    entropy_id = -np.sum(mean_probs_id * np.log(mean_probs_id + 1e-10), axis=1)  # Shape (20,)

    plot_id_vs_ood(entropy_ood, entropy_id)

if __name__ == "__main__":
    main()


