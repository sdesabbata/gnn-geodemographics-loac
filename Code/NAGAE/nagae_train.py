# Based on
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py [MIT license]
# https://github.com/vlukiyanov/pt-dec/blob/master/ptdec/cluster.py [MIT license]
# https://github.com/vlukiyanov/pt-dec/blob/master/ptdec/dec.py [MIT license]

# -------------------- #
#   Import libraries   #
# -------------------- #

# Base libraries
import os
import gc
import math
import time
# Local directories
from local_directories import *
# OS environment setup
os.chdir(this_repo_directory)
os.environ["USE_PYGEOS"] = "0"
# Data and plotting
import copy
import geopandas as gpd
import pandas as pd
import numpy as np
from math import ceil
import random
import itertools
import matplotlib.pyplot as plt
# Spatial
from pysal.lib import weights
from esda import Join_Counts_Local
# Machine Learning
from sklearn.cluster import KMeans
# Torch
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTNodeSampler
# NAGAE model
from nagae_models import *

# ----------------- #
#   General setup   #
# ----------------  #

setup_note = "NAGAE (v0.6) based on GAT with GraphSAINTNodeSampler"
random_seed = 456

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Run on CUDA")
else:
    print("Run on CPU")

# Set random seeds
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# ----------------- #
#   Load the data   #
# ----------------  #

# Load the census data (i.e., the node attributes)
census_data = pd.read_csv("data/input/Census-Data/GreaterLondon_2011_OAC_Raw_uVariables--zscores-with-colnames.csv")
census_data = census_data.set_index("OA11CD")

# Create a dictionary to map the OA index to the node index
mapping_OA_index = {index: i for i, index in enumerate(census_data.index.unique())}

# Reverse mapping
mapping_index_OA = {v: k for k, v in mapping_OA_index.items()}
mapping_index_OA_df = pd.DataFrame.from_dict(mapping_index_OA, orient="index", columns=["OA11CD"])
mapping_index_OA_df.head(n=15)

# Load the geometries of the spatial units
# spatial_data_file = "data/input/London-Map-OAs/ons-oa-geo-london_valid.geojson"
spatial_data_file = "data/input/London-Map-OAs/ons-oa-geo-london_valid_BNG.geojson"
spatial_data_df = gpd.read_file(spatial_data_file)


# Spatial graph definitions
def get_spatial_weights(spatial_df, spatial_graph_type):
    # Construct the spatial graph ...
    if spatial_graph_type == "queens":
        # ... using Queens graph
        spatial_weights = weights.contiguity.Queen.from_dataframe(spatial_df, ids="OA11CD")
    elif spatial_graph_type == "knn8":
        # ... using KNN(k=8) graph
        spatial_weights = weights.distance.KNN.from_dataframe(spatial_df, k=8, ids="OA11CD")
    elif spatial_graph_type == "mdt":
        # ... using a maximum distance threshold 2099 meters (min distance necessary for all OAs to be connected)
        spatial_weights = weights.distance.DistanceBand.from_dataframe(spatial_df, 2099, binary=True, ids="OA11CD")
    else:
        raise Exception("Sorry, I don't know that spatial graph type")
    return spatial_weights


# ------------------------------------- #
#   Define the design space to search   #
# ------------------------------------- #

# Set up the values for the designs to test
# # 621
# tests_id = "nagae_621"
# tests_spatial_graph = ["queens", "knn8", "mdt"]
# tests_layers_features_in = [167]
# tests_layers_features_prep = [[60], [167]]
# tests_layers_features_gatc = [[60, 60], [60, 60, 60, 60], [60, 60, 60, 60, 60, 60]]
# tests_layers_features_post = [[60], [60, 60]]
# tests_layers_features_attr = [[167], [60, 167], [60, 60, 167], [60, 167, 167]]
# tests_num_gat_heads = [2, 4, 8]
# tests_negative_slope = [0.02]
# tests_dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# tests_learning_rate = [0.0001]
# tests_epochs = [5000]
# tests_epochs_max_no_impr = [300]
# tests_batch_size = [256, 512, 1024, 2048]
# # 622
# tests_id = "nagae_622"
# tests_spatial_graph = ["queens", "knn8", "mdt"]
# tests_layers_features_in = [167]
# tests_layers_features_prep = [[60], [167]]
# tests_layers_features_gatc = [[60, 60], [60, 60, 60, 60]]
# tests_layers_features_post = [[60], [60, 60]]
# tests_layers_features_attr = [[167], [60, 167]]
# tests_num_gat_heads = [2, 4]
# tests_negative_slope = [0.02]
# tests_dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# tests_learning_rate = [0.0001]
# tests_epochs = [5000]
# tests_epochs_max_no_impr = [300]
# tests_batch_size = [1024, 2048]
# 623
tests_id = "nagae_623"
tests_spatial_graph = ["queens", "knn8", "mdt"]
tests_layers_features_in = [167]
tests_layers_features_prep = [[60]]
tests_layers_features_gatc = [[60, 60]]
tests_layers_features_post = [[60], [60, 60]]
tests_layers_features_attr = [[167], [60, 167]]
tests_num_gat_heads = [2, 4]
tests_negative_slope = [0.02]
tests_dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
tests_learning_rate = [0.0001]
tests_epochs = [5000]
tests_epochs_max_no_impr = [300]
tests_batch_size = [1024, 2048]

tests = list(itertools.product(
    tests_spatial_graph,
    tests_layers_features_in,
    tests_layers_features_prep,
    tests_layers_features_gatc,
    tests_layers_features_post,
    tests_layers_features_attr,
    tests_num_gat_heads,
    tests_negative_slope,
    tests_dropout,
    tests_learning_rate,
    tests_epochs,
    tests_epochs_max_no_impr,
    tests_batch_size
))

design_space = len(tests)
# tests = random.sample(tests, 100)
# tests = random.sample(tests, ceil(len(tests)/100))
print("Testing", len(tests), "of", design_space, "design combination possibilities\n\n")

os.mkdir(bulk_storage_directory + "__" + tests_id)
os.mkdir(bulk_storage_directory + "__" + tests_id + "/output")

results_timestamp_now = str(time.time()).replace(".", "_")
results_model_name = "results_nagae__"
results_file = open("data/output/" + results_model_name + results_timestamp_now + ".csv", "a")
results_file.writelines(
    "spatial_graph, layers_features_in, layers_features_encoder_prep, layers_features_encoder_gatc, " +
    "layers_features_encoder_post, layers_features_decoder_attr, num_gat_heads, negative_slope, dropout, " +
    "learning_rate, epochs, epochs_max_no_impr, batch_size, best_epoch, best_loss, best_model_auc, best_model_ap, best_model_mse, " +
    "output_labels_colname, sed_score, clustering_score, clustering_score_percentage\n")

for test in tests:

    # Reset random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # -------------------------- #
    #   Set the design to test   #
    # -------------------------- #

    test_spatial_graph = test[0]
    test_layers_features_in = test[1]
    test_layers_features_encoder_prep = test[2]
    test_layers_features_encoder_gatc = test[3]
    test_layers_features_encoder_post = test[4]
    test_layers_features_decoder_attr = test_layers_features_encoder_post[-1:] + test[5]
    test_num_gat_heads = test[6]
    test_negative_slope = test[7]
    test_dropout = test[8]
    test_learning_rate = test[9]
    test_epochs = test[10]
    test_epochs_max_no_impr = test[11]
    test_batch_size = test[12]

    test_as_string = [str(i) for i in test]
    test_string = ', '.join(test_as_string)

    # Output and log files info
    timestamp_now = str(time.time()).replace(".", "_")
    output_file_model_name = "nagae__" + test_spatial_graph + "_"
    output_labels_colname = output_file_model_name + timestamp_now
    output_file_path = (
            bulk_storage_directory + "__" + tests_id + "/output/" +
            output_file_model_name + timestamp_now +
            "__output.csv")
    output_final_clustering_map_path = (
            bulk_storage_directory + "__" + tests_id + "/output/" +
            output_file_model_name + timestamp_now +
            "___clusters")

    # Log a line to std output and log file
    def log_this(s, f):
        f.writelines(s + "\n")
        print(s)

    log_file = open("data/output/" + output_labels_colname + "__setup.txt", "a")
    log_this("\n\n---\n", log_file)
    log_this(test_string, log_file)
    log_this(timestamp_now, log_file)
    log_this(setup_note, log_file)
    log_this("Spatial graph: " + test_spatial_graph, log_file)
    log_this("NAGAE encoder features prep: " + " ".join([str(layer_size) for layer_size in test_layers_features_encoder_prep]), log_file)
    log_this("NAGAE encoder GAT conv: " + " ".join([str(layer_size) for layer_size in test_layers_features_encoder_gatc]), log_file)
    log_this("NAGAE encoder features post: " + " ".join([str(layer_size) for layer_size in test_layers_features_encoder_post]), log_file)
    log_this("NAGAE attribute decoder: " + " ".join([str(layer_size) for layer_size in test_layers_features_decoder_attr]), log_file)
    log_this("NAGAE num gat heads: " + str(test_num_gat_heads), log_file)
    log_this("NAGAE negative slope: " + str(test_negative_slope), log_file)
    log_this("NAGAE dropout: " + str(test_dropout), log_file)
    log_this("NAGAE learning rate: " + str(test_learning_rate), log_file)
    log_this("NAGAE epochs: " + str(test_epochs), log_file)
    log_this("NAGAE batch size: " + str(test_batch_size), log_file)
    log_this(output_labels_colname, log_file)
    log_this(output_file_path, log_file)
    log_this("", log_file)

    # Delete unnecessary variables
    del test_as_string, test_string

    # ---------------------------------- #
    #   Create Pytorch Geometric graph   #
    # ---------------------------------- #

    # Construct the spatial graph
    spatial_w = get_spatial_weights(spatial_data_df, test_spatial_graph)
    # Create dataframe of adjacency list
    spatial_w_adj_df = spatial_w.to_adjlist()
    # Add mapping index to adjacency table
    spatial_w_adj_df["focal_mapping"] = spatial_w_adj_df.apply(lambda row: mapping_OA_index[row["focal"]], axis=1)
    spatial_w_adj_df["neighbor_mapping"] = spatial_w_adj_df.apply(lambda row: mapping_OA_index[row["neighbor"]], axis=1)

    # Create node attributes tensor
    geodemo_nodes_attr = torch.tensor(
        census_data.to_numpy(),
        dtype=torch.float)
    # Create spatial edges tensor
    geodemo_edges_idx = torch.tensor(
        spatial_w_adj_df[["focal_mapping", "neighbor_mapping"]].to_numpy().transpose(),
        dtype=torch.long)
    # Create graph
    geodemo_graph = Data(x=geodemo_nodes_attr, edge_index=geodemo_edges_idx)

    # Set up the node sampler to generate the minibatches
    graph_batch_loader = GraphSAINTNodeSampler(
        geodemo_graph,
        batch_size=test_batch_size,
        num_steps=ceil(25053 / test_batch_size),
        # num_steps=100,
        sample_coverage=100
    )

    # -------------------------- #
    #   Define the NAGAE model   #
    # -------------------------- #

    # Encoder
    nagae_encoder = GATEncoder(
        layers_features_in=test_layers_features_in,
        layers_features_prep=test_layers_features_encoder_prep,
        layers_features_gatc=test_layers_features_encoder_gatc,
        layers_features_post=test_layers_features_encoder_post,
        num_gat_heads=test_num_gat_heads,
        negative_slope=test_negative_slope,
        dropout=test_dropout
    )
    # Decoder (attributes)
    nagae_attribute_decoder = AttributeDecoder(
        layers_features=test_layers_features_decoder_attr,
        negative_slope=test_negative_slope
    )
    # NAGAE model
    nagae_model = NAGAE(nagae_encoder, nagae_attribute_decoder).to(device)
    # Optimizer
    nagae_encoder_optimizer = torch.optim.AdamW(nagae_model.parameters(), lr=test_learning_rate)

    print("\n")
    print(nagae_model)
    print("\n")

    # Final number of hidden features used for clustering
    geodemo_num_of_features = test_layers_features_encoder_post[-1]
    # Delete unnecessary variables
    del test_spatial_graph, test_layers_features_in, test_layers_features_encoder_prep, \
        test_layers_features_encoder_gatc, test_layers_features_encoder_post, test_layers_features_decoder_attr, \
        test_num_gat_heads, test_negative_slope, test_dropout, test_learning_rate, test_batch_size

    # ------------------------- #
    #   Train the NAGAE model   #
    # ------------------------- #

    # Set up best model and related info
    nagae_best_model = None
    nagae_best_loss = math.inf
    nagae_best_epoch = 0

    for epoch in range(1, test_epochs + 1):

        # Exit training if there has been no improvement in the past 100 epochs
        if (epoch - nagae_best_epoch) > test_epochs_max_no_impr:
            print("\tThere have been {:03d} epochs without improvement, exiting training".format((epoch - nagae_best_epoch)))
            break

        print("Epoch: {:03d}".format(epoch))
        epoch_loss = 0
        # Train mode
        nagae_model.train()

        batch_index = 0
        batches_loss_str = "\tBatch loss log: "
        for batch_sample in graph_batch_loader:
            batch_sample = batch_sample.to(device)
            batch_index += 1
            # zero the parameter gradients
            nagae_encoder_optimizer.zero_grad()
            # forward pass: encode, decode and return loss
            batch_loss = nagae_model(batch_sample.x, batch_sample.edge_index)
            epoch_loss += float(batch_loss)
            # backpropagation
            batch_loss.backward()
            # optimise
            nagae_encoder_optimizer.step()
            # print batch loss
            # print("\tBatch {:03d}: Attribute loss: {:.4f}".format(batch_index, float(loss)))
            batches_loss_str += "b{:03d}={:.4f} ".format(batch_index, float(batch_loss))
            # Delete unnecessary variables
            del batch_sample, batch_loss
        # Print batch loss summary
        print(batches_loss_str)
        # Delete unnecessary variables
        del batches_loss_str

        # Calculate average loss
        epoch_loss = epoch_loss / batch_index
        print("\tEpoch train attribute loss (average): {:.4f}".format(epoch_loss))
        # Update best model if necessary
        if epoch_loss < nagae_best_loss:
            nagae_best_model = copy.deepcopy(nagae_model)
            nagae_best_loss = epoch_loss
            nagae_best_epoch = epoch
        # Delete unnecessary variables
        del epoch_loss, batch_index

    print("\nThe best model was trained in epoch {:03d}, with train attribute loss (average): {:.4f}\n".format(nagae_best_epoch, nagae_best_loss))
    # Delete unnecessary variables
    del test_epochs, nagae_model

    # Final test for the best model and clustering procedure
    with torch.no_grad():

        # ------------------------ #
        #   Test the NAGAE model   #
        # ------------------------ #

        # Evaluation mode
        nagae_best_model.eval()
        # AUC, AP and MSE totals
        nagae_best_model_auc, nagae_best_model_ap, nagae_best_model_mse = 0, 0, 0
        batch_index = 0
        for batch_sample in graph_batch_loader:
            batch_sample = batch_sample.to(device)
            batch_index += 1
            # test
            batch_mse, batch_auc, batch_ap = nagae_best_model.test(batch_sample)
            # sum AUC, AP and MSE to totals
            nagae_best_model_mse += batch_mse
            if nagae_best_model_auc is None or nagae_best_model_ap is None:
                # if there have been issues with the edge recon test, take average so far
                nagae_best_model_auc += (nagae_best_model_auc / batch_index)
                nagae_best_model_ap += (nagae_best_model_ap / batch_index)
            else:
                # ... otherwise sum obtained values to the total
                nagae_best_model_auc += batch_auc
                nagae_best_model_ap += batch_ap
            # Delete unnecessary variables
            del batch_sample, batch_mse, batch_auc, batch_ap
        # Calculate AUC, AP and MSE averages
        nagae_best_model_mse = nagae_best_model_mse / batch_index
        nagae_best_model_auc = nagae_best_model_auc / batch_index
        nagae_best_model_ap = nagae_best_model_ap / batch_index
        # Log results
        if nagae_best_model_auc is None or nagae_best_model_ap is None:
            log_this("Test: Attributes MSE: {:.4f} (Note: something went wrong with edge reconstruction )".format(nagae_best_model_mse), log_file)
        else:
            log_this("Test: Attributes MSE: {:.4f}, Edges AUC: {:.4f}, Edges AP: {:.4f}".format(nagae_best_model_mse, nagae_best_model_auc, nagae_best_model_ap), log_file)
        # Delete unnecessary variables
        del batch_index

        # ------------------------------#
        #   Final clustering process   #
        # ------------------------------#

        # Move model to CPU for computing final values to be used for KMeans
        # that's due to 2 reasons:
        # - Some models are really big and GPU RAM is limited on HPC. The code above uses mini batches,
        #   but we need the full graph here
        # - When looping through different test settings, need to keep geodemo_graph on CPU for the loader

        nagae_best_model.to("cpu")
        nagae_best_model.eval()

        # Encoding the entire graph can require a lot of memory
        # 1) enact explicit garbage collection
        # 2) use exception so not to stop the process
        gc.collect()

        try:
            nagae_best_model_encoded = nagae_best_model.encode(geodemo_graph.x, geodemo_graph.edge_index)
            nagae_best_model_features = nagae_best_model_encoded.cpu().detach().numpy()

            # Clustering model
            geodemo_kmeans = KMeans(n_clusters=8, random_state=456, n_init=100, tol=0.0001, max_iter=500, verbose=1)
            nagae_best_model_kmeans_predicted = geodemo_kmeans.fit_predict(nagae_best_model_features)

            # Combine all into one table
            geodemo_df = pd.concat([
                mapping_index_OA_df,
                pd.DataFrame(nagae_best_model_features, columns=["nagae_" + str(i + 1) for i in range(geodemo_num_of_features)]),
                pd.DataFrame(nagae_best_model_kmeans_predicted, columns=[output_labels_colname]),
            ], axis=1)
            geodemo_df[output_labels_colname] = geodemo_df[output_labels_colname].astype("category")
            geodemo_df.head()

            # Plot
            pd.merge(spatial_data_df, geodemo_df, on="OA11CD").plot(column=output_labels_colname, cmap="Set1",
                                                                    figsize=(20, 20), legend=True)
            plt.savefig(output_final_clustering_map_path + ".png")
            plt.close()

            # Save output
            geodemo_df.to_csv(output_file_path, index=False)

            # Calculate Square Euclidean Distance (SED) score
            # Calculate centroids first
            sed_centroids = census_data.groupby(nagae_best_model_kmeans_predicted).mean()

            # Calculate squared euclidean distance from centroid
            def sed(row):
                centroid = sed_centroids.loc[row[-1]]
                return np.sum((row[:-1] - centroid) ** 2)


            census_data_with_labels = census_data.copy()
            census_data_with_labels["labels"] = nagae_best_model_kmeans_predicted
            sed_distances = census_data_with_labels.apply(sed, axis=1)
            # Calculate total SED
            sed_score = sed_distances.mean()
            log_this("The Square Euclidean Distance (SED 167 z-scores) score is: {:.4f}".format(sed_score), log_file)

            # Calculate clustering score
            # (this is just a preliminary score based on proximity, not the final one used in the paper)
            jcl = Join_Counts_Local(get_spatial_weights(spatial_data_df, "queens"))
            clustering_score = 0
            for i in np.unique(nagae_best_model_kmeans_predicted):
                jcl_fit = jcl.fit((nagae_best_model_kmeans_predicted == i).astype(int))
                jcl_fit_clustered = (jcl_fit.LJC > 0).astype(int).sum()
                clustering_score += jcl_fit_clustered
                log_this("Clustered in {:.0f}: {:.0f}".format(i, jcl_fit_clustered), log_file)
                # Delete unnecessary variables
                del jcl_fit, jcl_fit_clustered
            # join counts as percentage of the number of OAs
            clustering_score_percentage = (clustering_score / nagae_best_model_kmeans_predicted.size) * 100.0
            log_this("The clustering (based on join counts) score is: {:.2f}% (count: {:.0f})".format(clustering_score_percentage, clustering_score), log_file)

            # Write results to file
            test_result = test + (
                nagae_best_epoch, nagae_best_loss, nagae_best_model_auc, nagae_best_model_ap, nagae_best_model_mse,
                output_labels_colname, sed_score, clustering_score, clustering_score_percentage)
            test_result_as_string = [str(i) for i in test_result]
            test_result_string = ', '.join(test_result_as_string)
            log_this(test_result_string, log_file)
            results_file.writelines(test_result_string + "\n")

            del nagae_best_model_encoded, nagae_best_model_features
            del geodemo_kmeans, nagae_best_model_kmeans_predicted, geodemo_df
            del sed_centroids, census_data_with_labels, sed_distances, sed_score
            del jcl, clustering_score, clustering_score_percentage
            del test_result, test_result_as_string, test_result_string

        except RuntimeError as err:
            print("\nHandled RuntimeError")
            print(err)
            print("\n\n")
            output_labels_colname = "NA"
            sed_score = "NA"
            clustering_score = "NA"
            clustering_score_percentage = "NA"

            # Write results to file
            test_result = test + (
                nagae_best_epoch, nagae_best_loss, nagae_best_model_auc, nagae_best_model_ap, nagae_best_model_mse,
                output_labels_colname, sed_score, clustering_score, clustering_score_percentage)
            test_result_as_string = [str(i) for i in test_result]
            test_result_string = ', '.join(test_result_as_string)
            log_this(test_result_string, log_file)
            results_file.writelines(test_result_string + "\n")

            del sed_score, clustering_score, clustering_score_percentage
            del test_result, test_result_as_string, test_result_string

    # Close and delete log file
    log_file.close()
    del log_file

    # Delete unnecessary variables
    del timestamp_now, output_file_model_name, output_labels_colname, output_file_path, output_final_clustering_map_path
    del spatial_w, spatial_w_adj_df, geodemo_nodes_attr, geodemo_edges_idx, geodemo_graph, graph_batch_loader
    del nagae_encoder, nagae_attribute_decoder, nagae_encoder_optimizer, geodemo_num_of_features
    del nagae_best_model, nagae_best_loss, nagae_best_epoch, nagae_best_model_auc, nagae_best_model_ap, nagae_best_model_mse

results_file.close()
