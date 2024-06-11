import torch
import torch.nn.functional as F
from framework.ccnet.cooperative_network import CooperativeNetwork
from tools.report import calculate_test_results

class CCNetAnalysis:
    def analyze_causal_explanation(ccnet: CooperativeNetwork, observation, label):
        task_type, label_size, label_scale = ccnet.task_type, ccnet.label_size, ccnet.label_scale
        explain_size = ccnet.explain_size
        explain = ccnet.explain(observation)
        
        generated_x = ccnet.producer(label, explain)
        
        generated_explain = ccnet.explain(generated_x)
    
        metrics = calculate_test_results(explain, generated_explain, task_type = 'regression', label_size=explain_size, label_scale = None) 
        return metrics

    def analyze_causal_inference(ccnet: CooperativeNetwork, observation, label):
        task_type, label_size, label_scale = ccnet.task_type, ccnet.label_size, ccnet.label_scale
        
        explain = ccnet.explain(observation)
        
        generated_x = ccnet.produce(label, explain)
        
        generated_infer = ccnet.infer(generated_x)
    
        metrics = calculate_test_results(label, generated_infer, task_type, label_size, label_scale)
        return metrics

    def analyze_conditional_independence(ccnet: CooperativeNetwork, observation, label, learned_model):
        task_type, label_size, label_scale = ccnet.task_type, ccnet.label_size, ccnet.label_scale
        
        # Generate explanations from input observations
        explaination = ccnet.explain(observation)
        
        # Determine the batch size
        batch_size = observation.shape[0]
        
        # Randomize the shift amount within 1 to batch_size - 1
        shift_amount = torch.randint(1, batch_size, (1,)).item()
        
        # Shift the explanations by the random amount
        shifted_indices = (torch.arange(batch_size) + shift_amount) % batch_size
        explain_changed = explaination[shifted_indices]
        
        # Generate new X using label and shifted explanations
        generated_x = ccnet.produce(label, explain_changed)
        
        # Predict Y using the learned model on generated X
        prediction_y = learned_model(generated_x)
        
        # Calculate metrics to compare predictions
        metrics = calculate_test_results(prediction_y, label, task_type, label_size, label_scale)
        
        return metrics

    def analyze_causal_generation(ccnet: CooperativeNetwork, observation, label):
        task_type, label_size, label_scale = ccnet.task_type, ccnet.label_size, ccnet.label_scale
        
        explain = ccnet.explain(observation)
        
        generated_x = ccnet.produce(label, explain)
    
        metrics = calculate_test_results(generated_x, observation, task_type, label_size, label_scale)
        return metrics

    def analyze_causal_reconstruction(ccnet: CooperativeNetwork, observation, label):
        task_type, label_size, label_scale = ccnet.task_type, ccnet.label_size, ccnet.label_scale
        
        explain = ccnet.explain(observation)
        
        inferred_y = ccnet.reason(observation, explain)

        reconstructed_x = ccnet.produce(inferred_y, explain)
    
        metrics = calculate_test_results(reconstructed_x, observation, task_type, label_size, label_scale)
        return metrics
        