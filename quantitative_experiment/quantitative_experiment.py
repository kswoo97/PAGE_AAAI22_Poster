import numpy as np
import pandas as pd
import torch
import torch_geometric
device = torch.device("cpu")
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
from scipy import stats
import argparse

# Utility functions
from graph_classification_utils import *

# GNN models
from graph_classification_gnns import *

# Explainer
from graph_classification_prototype_explainer import *

# Benchmark
from graph_classification_XGNN import *

def consistency_testing(testing_data) :
    # Generate data
    print("Started consistency testing of {0}".format(testing_data))
    if testing_data == "ba_house" :
        #House data
        ba_train_model = ba_house_generator(max_n=10,
                                            min_n=5,
                                            edge_rate=2,
                                            r_seed=0)
        ba_test_model = ba_house_generator(max_n=10,
                                           min_n=5,
                                           edge_rate=2,
                                           r_seed=2000)
        ba_train_model.dataset_generator(num_graph=2000)
        ba_train_module = DataLoader(ba_train_model.data_list, batch_size=10)
        ba_test_model.dataset_generator(num_graph=1000)
        ba_test_data = ba_test_model.data_list

        # 1. BA-house
        data_type = "BA_house"
        house_model_performance = []
        house_explanation_performance = []
        house_XGNN_performance = []
        first_layer = [4, 8, 16, 32, 64]
        second_layer = [4, 8, 16, 32, 64]
        count = 0
        for f1 in first_layer :
            for f2 in second_layer :
                count += 1
                print("Currently in {0}/25 process".format(count))
                house_gnn = GCN_Conv(dataset = ba_train_model.data_list[0], latent_dim = [f1, f2])
                house_gnn = ba_house_training(model=house_gnn,
                                              data_module=ba_train_module,
                                              test_data=ba_test_data,
                                              device="cpu", epochs=30)
                house_gnn.eval()
                house_model_performance.append(ba_house_class_evaluator(house_gnn, ba_test_data, "cpu"))
                house_explainer = prototype_explanation(gnn_model=house_gnn,
                                                        dataset=ba_train_model.data_list,
                                                        data_name="ba_house")
                resulting_prototypes = house_explainer.generate_prototype(label=1,
                                                                          k=3, n_components=2,
                                                                          n_iter=10,
                                                                          cluster_index=0,
                                                                          max_epochs=100)
                house_explanation_performance.append(house_gnn(data=resulting_prototypes[0],
                                                               training_with_batch=False)[0].detach().item())
                house_XGNN = XGNN_model(dataset=ba_train_model.data_list[0],
                                        candidates=torch.diag(torch.ones(3)),
                                        data_type="ba_house",
                                        gcn_latent_dim=[8, 16])
                trained_XGNN, final_graph = train_XGNN(explainer=house_XGNN,
                                                       gnn_model=house_gnn,
                                                       initial_n=5,
                                                       max_node_n=15,
                                                       lambda_1=1,
                                                       lambda_2=2,
                                                       m=10,
                                                       label=1,
                                                       init_type=0)
                house_XGNN_performance.append(house_gnn(data=final_graph,
                                                        training_with_batch=False)[0].detach().item())
        return (house_model_performance, house_explanation_performance, house_XGNN_performance)

    # Solubility data
    elif testing_data == "solubility" :
        solubility_data = solubility_data_generator(path="solubility_original_data.csv")
        solubility_train_data = solubility_data[:500]
        solubility_test_data = solubility_data[500:]
        solubility_train_module = DataLoader(solubility_train_data, batch_size=5)
        # 2. Solubility
        solubility_model_performance = []
        solubility_explanation_performance = []
        solubility_XGNN_performance = []
        first_layer = [8, 16, 32, 64]
        second_layer = [8, 16, 32, 64]
        third_layer = [8, 16, 32, 64]
        count = 0
        for f1 in first_layer:
            for f2 in second_layer :
                for f3 in third_layer :
                    count += 1
                    print("Currently in {0}/64 process".format(count))
                    solubility_gnn = GCN_Conv(dataset=solubility_data[0], latent_dim=[f1, f2, f3])
                    solubility_gnn = solubility_training(model=solubility_gnn,
                                                       data_module=solubility_train_module,
                                                       test_data=solubility_test_data,
                                                       device="cpu", epochs=300)
                    solubility_model_performance.append(solubility_class_evaluator(solubility_gnn, solubility_test_data, "cpu"))
                    solubility_explainer = prototype_explanation(gnn_model=solubility_gnn,
                                                                 dataset=solubility_data,
                                                                 data_name="solubility")
                    resulting_prototypes = solubility_explainer.generate_prototype(label=0,
                                                                                   k=3, n_components=3,
                                                                                   n_iter=5,
                                                                                   cluster_index=0,
                                                                                   max_epochs=100)
                    solubility_explanation_performance.append(1 - solubility_gnn(data=resulting_prototypes[0],
                                                                                 training_with_batch=False)[0].detach().item())
                    XGNN = XGNN_model(dataset=solubility_data[0],
                                      candidates=torch.diag(torch.ones(9)),
                                      data_type="solubility",
                                      gcn_latent_dim=[8, 16])

                    trained_XGNN, final_graph = train_XGNN(explainer=XGNN,
                                                           gnn_model=solubility_gnn,
                                                           initial_n=5,
                                                           max_node_n=15,
                                                           lambda_1=1,
                                                           lambda_2=2,
                                                           m=10,
                                                           label=0,
                                                           init_type=0)
                    solubility_XGNN_performance.append(solubility_gnn(data=final_graph,
                                                                      training_with_batch=False)[0].detach().item())

        return (solubility_model_performance, solubility_explanation_performance, solubility_XGNN_performance)

def faithfulness_testing(testing_data) :
    # Generate data
    print("Started faithfulness testing of {0}".format(testing_data))
    if testing_data == "ba_house" :
        #House data
        ba_train_model = ba_house_generator(max_n=10,
                                            min_n=5,
                                            edge_rate=2,
                                            r_seed=0)
        ba_test_model = ba_house_generator(max_n=10,
                                           min_n=5,
                                           edge_rate=2,
                                           r_seed=2000)
        ba_train_model.dataset_generator(num_graph=2000)
        ba_train_module = DataLoader(ba_train_model.data_list, batch_size=10)
        ba_test_model.dataset_generator(num_graph=1000)
        ba_test_data = ba_test_model.data_list

        # 1. BA-house
        data_type = "BA_house"
        house_model_performance = []
        house_explanation_performance = []
        house_XGNN_performance = []
        house_gnn = GCN_Conv(dataset = ba_train_model.data_list[0], latent_dim = [32, 32])
        house_optim = torch.optim.Adam(house_gnn.parameters(), lr = 0.001, weight_decay=5e-4)
        criterion = torch.nn.BCELoss()
        house_gnn.train()
        for i in tqdm(range(30)) : # Performing 30 epochs
            house_gnn.train()
            for data in ba_train_module :
                house_optim.zero_grad()
                out = house_gnn(data = data, training_with_batch = True)
                loss = criterion(out.view(-1), data.y)
                loss.backward()
                house_optim.step()
            house_gnn.eval()
            house_model_performance.append(ba_house_class_evaluator(house_gnn, ba_test_data, "cpu"))
            house_explainer = prototype_explanation(gnn_model=house_gnn,
                                                    dataset=ba_train_model.data_list,
                                                    data_name="ba_house")
            resulting_prototypes = house_explainer.generate_prototype(label=1,
                                                              k=3, n_components=2,
                                                              n_iter=4,
                                                              cluster_index=0,
                                                              max_epochs=40)
            house_explanation_performance.append(house_gnn(data = resulting_prototypes[0],
                                                     training_with_batch = False)[0].detach().item())
            house_XGNN = XGNN_model(dataset = ba_train_model.data_list[0],
                               candidates = torch.diag(torch.ones(3)),
                               data_type = "ba_house",
                               gcn_latent_dim = [8, 16])
            trained_XGNN, final_graph = train_XGNN(explainer=house_XGNN,
                                                   gnn_model=house_gnn,
                                                   initial_n=5,
                                                   max_node_n=15,
                                                   lambda_1=1,
                                                   lambda_2=2,
                                                   m=10,
                                                   label=1,
                                                   init_type=0)
            house_XGNN_performance.append(house_gnn(data = final_graph,
                                                     training_with_batch = False)[0].detach().item())

        return (house_model_performance, house_explanation_performance, house_XGNN_performance)

    # 2. Solubility
    elif testing_data == "solubility" :
        # Solubility data
        solubility_data = solubility_data_generator(path="solubility_original_data.csv")
        solubility_train_data = solubility_data[:500]
        solubility_test_data = solubility_data[500:]
        solubility_train_module = DataLoader(solubility_train_data, batch_size=5)
        solubility_model_performance = []
        solubility_explanation_performance = []
        solubility_XGNN_performance = []
        solubility_gnn = GCN_Conv(dataset=solubility_train_data[0], latent_dim=[32, 32, 32])
        solubility_optim = torch.optim.Adam(solubility_gnn.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.BCELoss()
        solubility_gnn.train()
        for i in tqdm(range(300)) :
            solubility_gnn.train()
            for data in solubility_train_module :
                solubility_optim.zero_grad()
                out = solubility_gnn(data=data, training_with_batch=True)
                loss = criterion(out.view(-1), data.y)
                loss.backward()
                solubility_optim.step()
            solubility_gnn.eval()
            solubility_model_performance.append(solubility_class_evaluator(solubility_gnn, solubility_test_data, "cpu"))
            solubility_explainer = prototype_explanation(gnn_model=solubility_gnn,
                                                         dataset=solubility_data,
                                                         data_name="solubility")
            resulting_prototypes = solubility_explainer.generate_prototype(label=0,
                                                                   k=3, n_components=3,
                                                                   n_iter=6,
                                                                   cluster_index=0,
                                                                   max_epochs=50)
            solubility_explanation_performance.append(1 - solubility_gnn(data = resulting_prototypes[0],
                                                                         training_with_batch = False)[0].detach().item())
            XGNN = XGNN_model(dataset=solubility_data[0],
                              candidates=torch.diag(torch.ones(9)),
                              data_type="solubility",
                              gcn_latent_dim=[8, 16])

            trained_XGNN, final_graph = train_XGNN(explainer=XGNN,
                                                   gnn_model=solubility_gnn,
                                                   initial_n=5,
                                                   max_node_n=15,
                                                   lambda_1=1,
                                                   lambda_2=2,
                                                   m=10,
                                                   label=0,
                                                   init_type=0)
            solubility_XGNN_performance.append(solubility_gnn(data=final_graph,
                                                    training_with_batch=False)[0].detach().item())

        return (solubility_model_performance, solubility_explanation_performance, solubility_XGNN_performance)

def record_final_result() :
    house_c_1, house_c_2, house_c_3 = consistency_testing("ba_house")
    solub_c_1, solub_c_2, solub_c_3 = consistency_testing("solubility")
    house_f_1, house_f_2, house_f_3 = faithfulness_testing("ba_house")
    solub_f_1, solub_f_2, solub_f_3 = faithfulness_testing("solubility")
    print("=============== Experiment done. Saving result... ===============")
    ba_page_consist = np.std(house_c_2) ; ba_xgnn_consist = np.std(house_c_3)
    sol_page_consist = np.std(solub_c_2); sol_xgnn_consist = np.std(solub_c_3)
    ba_page_faith = stats.kendalltau(house_f_1, house_f_2)[0]
    ba_xgnn_faith = stats.kendalltau(house_f_1, house_f_3)[0]
    sol_page_faith = stats.kendalltau(solub_f_1, solub_f_2)[0]
    sol_xgnn_faith = stats.kendalltau(solub_f_1, solub_f_3)[0]
    resulting_table = pd.DataFrame({"page_consist" : [ba_page_consist, sol_page_consist],
                  "xgnn_consist" : [ba_xgnn_consist, sol_xgnn_consist],
                  "page_faith" : [ba_page_faith, sol_page_faith],
                  "xgnn_faith" : [ba_xgnn_faith, sol_xgnn_faith]},
                                   index = ["ba_house", "solubility"])
    return resulting_table

if __name__ == "__main__" :

    results = record_final_result()
    results.to_csv("final_result.csv")