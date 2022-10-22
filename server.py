import flwr 
from typing import  Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from functools import reduce
import numpy as np
import pickle
import warnings
from torch.utils.tensorboard import SummaryWriter
import copy

from dsfl import alastor, Flatten

warnings.filterwarnings("ignore", category=UserWarning)

class History():
    def __init__(self):
        self.list = []
        self.error = []
        self.globals = []
        self.round = 0
        
    def update(self, weighted_weights):
        copied_list=copy.deepcopy(weighted_weights)
        self.list.append(copied_list)
        self.round +=1
        
        print('\n', "Gobal Round", self.round)
        
    def updateError(self, errors):
        self.error.append(errors[:])
        
    def updateGlobal(self, globmodel):
        self.globals.append(globmodel[:])
        
    def Len(self):
        
        print(len(self.list))
    
    each_round_weighted_weights=[]
   
    

def aggregateNew(results: List[Tuple[Weights, int]],client_id) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * 1 for layer in weights] for weights, num_examples in results
    ]
    
    #sort and match the id with weights
    weighted_weights = [weights for _,weights in sorted(zip(client_id,weighted_weights),key=lambda pair: pair[0])]

            
    # print(weighted_weights)
    # print(len(weighted_weights))
    # [layer * num_examples for layer in weights] for weights, num_examples in results
# ]


    history.update(weighted_weights)
    with open("test", "wb") as fp:
        pickle.dump(history.list, fp)
    # Compute average weights of each layer
    
    """ Saved File:
        First Index: Round 
        Second Index: User
        Third Index: Layer
    """

    
    """Alastor's function"""
    weighted_weights_accu=copy.deepcopy(weighted_weights) 
    weighted_weights=alastor(weighted_weights_accu, history)
 

    """Aggregate"""
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / number_of_users #num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    
    return weights_prime



if __name__ == "__main__":

    number_of_users=10
    
    history=History()
    writer = SummaryWriter(comment= " MNIST - NIID - Dyn - alpha = 100, gamma = 10 - Test redo 0") 
    # writer = SummaryWriter(comment= "Best - MNIST - Non-iid") 
    # writer = SummaryWriter(comment= "Trying New Things")

    #Extend class FedAVG
    class FedComp(FedAvg):
        
        """Save and graph the aggregated loss and accuraies"""
        def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
        ) -> Optional[float]:
            """Aggregate evaluation losses using weighted average."""
            if not results:
                return None
    
            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            loss = [r.loss * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
    
            # Aggregate and print custom metric
            accuracy_aggregated = sum(accuracies) / sum(examples)
            loss_aggregated = sum(loss) / sum(examples)
            print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
            print(f"Round {rnd} loss aggregated from client results: {loss_aggregated}")
            writer.add_scalar('accuracy_aggregated', accuracy_aggregated, rnd)
            writer.add_scalar('loss_aggregated', loss_aggregated, rnd)
            
            # Call aggregate_evaluate from base class (FedAvg)
            return super().aggregate_evaluate(rnd, results, failures)
                
        def aggregate_fit(
                self,
                rnd: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[BaseException],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                """Aggregate fit results using weighted average."""
                client_id=[int(client.cid[11:]) for client, _ in results]
                if not results:
                    return None, {}
                # Do not aggregate if there are failures and failures are not accepted
                if not self.accept_failures and failures:
                    return None, {}
                # Convert results
                weights_results = [
                    (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                    for client, fit_res in results
                ]
                return weights_to_parameters(aggregateNew(weights_results,client_id)), {}
    

    # Set the initial model for reproducability
    infile = open('initial_global_model_MNIST','rb')
    initial_model = pickle.load(infile)
    
    history.updateGlobal(Flatten(initial_model[0][0])[0])
    
    history.updateError([[0]*len(Flatten(initial_model[0][0])[0]) for _ in range(number_of_users)])
    
    # Define strategy and 
    strategy = FedComp(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=number_of_users,  # Minimum number of clients to be sampled for the next round
        min_available_clients=number_of_users,  # Minimum number of clients that need to be connected to the server before a training round can start
        initial_parameters=initial_model[0][0]

    )


    
    
    # Start server
    flwr.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 300},
        strategy=strategy,
    )