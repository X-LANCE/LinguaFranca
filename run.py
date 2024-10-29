import torch
from datasets import load_dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
import seaborn as sns
from lib.eval import eval_ppl_multilingual
import json
import lm_eval 
import csv


def remove_hs_hooks_from_llama(model):
    for layer in model.model.layers:
        layer.mlp.down_proj._forward_hooks.clear()


def remove_hs_hooks_from_bloom(model):
    for layer in model.transformer.h:
        layer.mlp.gelu_impl._forward_hooks.clear()


class SaveActivationHookLlama:
    def __init__(self, hidden_dim, layer_num):

        self.results = []
        self.layer_num = layer_num
        self.num_token_generated = 0
        self.hidden_dim = hidden_dim
        self.final_results = torch.zeros(self.hidden_dim * self.layer_num)

    def __call__(self, module, input, output):
        # Assuming output is the result of the linear layer (gate)
        # For example, if output is (batch_size, sequence_length, 8)
        # you can reshape it to (batch_size * sequence_length, 8)
        
        output_reshaped = input[0][0].sum(dim=0).detach().to('cpu')
        
        # Save the results for this layer
        self.results.append(output_reshaped)

        if len(self.results) >= self.layer_num:
          self.final_results = (self.final_results * self.num_token_generated + self.get_concatenated_results()) / (self.num_token_generated + len(input[0]))

          assert len(self.final_results) == self.hidden_dim * self.layer_num

          self.num_token_generated = self.num_token_generated + len(input[0])
          self.results = []

    def get_concatenated_results(self):
        # Concatenate the results for all layers into a single tensor
        concatenated_results = torch.cat(self.results, dim=0)

        return concatenated_results

    def initialize(self):
        self.results = []
        self.num_token_generated = 0
        self.final_results = torch.zeros(self.hidden_dim * self.layer_num)



class SaveActivationHookBloom:
    def __init__(self, hidden_dim, layer_num):

        self.results = []
        self.layer_num = layer_num
        self.num_token_generated = 0
        self.hidden_dim = hidden_dim
        self.final_results = torch.zeros(self.hidden_dim * self.layer_num)

    def __call__(self, module, input, output):
        # Assuming output is the result of the linear layer (gate)
        # For example, if output is (batch_size, sequence_length, 8)
        # you can reshape it to (batch_size * sequence_length, 8)
        
        output_reshaped = (torch.sum(torch.sum(output, 0)/output.shape[0], 0)/output.shape[1]).detach().to('cpu')
        
        # Save the results for this layer
        self.results.append(output_reshaped)

        if len(self.results) >= self.layer_num:
          self.final_results = (self.final_results * self.num_token_generated + self.get_concatenated_results()) / (self.num_token_generated + len(output))

          assert len(self.final_results) == self.hidden_dim * self.layer_num

          self.num_token_generated = self.num_token_generated + len(output)
          self.results = []

    def get_concatenated_results(self):
        # Concatenate the results for all layers into a single tensor
        concatenated_results = torch.cat(self.results, dim=0)

        return concatenated_results

    def initialize(self):
        self.results = []
        self.num_token_generated = 0
        self.final_results = torch.zeros(self.hidden_dim * self.layer_num)

def generate_hidden_states_bloom(sentence_list, model, device, tokenizer, hs_dim = 16384, tokenize = False):

  outputs = []
  hook = SaveActivationHookBloom(hs_dim, len(model.transformer.h))

  for layer in model.transformer.h:
    layer.mlp.gelu_impl.register_forward_hook(hook)

  for sentence in tqdm(sentence_list):
    if tokenize:
        input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    else:
        input_ids = sentence
    result = model.generate(inputs=input_ids, max_new_tokens=1)
    ## decode the result
    
    outputs.append(hook.final_results)
    hook.initialize()
    

  remove_hs_hooks_from_bloom(model)

  return outputs


def render_combined_sentences(index, lang, dataset):

    text = ""
    for i in range(index*1, index*1+1):
        text += dataset["train"][i]["translation"][lang]

    return text


def generate_hidden_states_llama(sentence_list, model, device, tokenizer, hs_dim = 14336, tokenize = False):

  outputs = []
  hook = SaveActivationHookLlama(hs_dim, len(model.model.layers))

  for layer in model.model.layers:
    layer.mlp.down_proj.register_forward_hook(hook)

  for sentence in tqdm(sentence_list):
    if tokenize:
        input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    else:
        input_ids = sentence
    result = model.generate(inputs=input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)

    outputs.append(hook.final_results)
    hook.initialize()

  remove_hs_hooks_from_llama(model)

  return outputs


def similarity_by_block(cosine_similarity_matrix, block_size):

    nrows, ncols = cosine_similarity_matrix.shape
    assert nrows%block_size==0 and ncols%block_size==0

    # Calculate the number of blocks along each dimension
    num_blocks_row = np.ceil(nrows / block_size).astype(int)
    num_blocks_col = np.ceil(ncols / block_size).astype(int)

    # Initialize the averaged matrix
    averaged_matrix = np.zeros((num_blocks_row, num_blocks_col))

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            # Calculate the row/column indices for the current block
            row_start = i * block_size
            row_end = min((i + 1) * block_size, nrows)
            col_start = j * block_size
            col_end = min((j + 1) * block_size, ncols)

            # Extract the block and calculate its average
            block = cosine_similarity_matrix[row_start:row_end, col_start:col_end].copy()

            # print(f"Block {i}, {j}")
            # compare_similarity_diagnal(block)

            # mask diagonal
            mask = np.ones((block_size, block_size), dtype=bool)
            np.fill_diagonal(mask, False)
            non_diagonal_elements = block[mask]
            average = np.mean(non_diagonal_elements)
            averaged_matrix[i, j] = average

    return averaged_matrix

def compare_similarity_diagnal(matrix):
    
    diagonal_mean = np.mean(np.diag(matrix))
    size = matrix.shape[0]
    off_diagonal_mean = (np.sum(matrix) - np.sum(np.diag(matrix))) / (size*size - size)
    print(f"The score is {diagonal_mean - off_diagonal_mean}")

    return diagonal_mean, off_diagonal_mean

def rearrange_by_meaning(loaded_tensor_list, sample_size_per_language):

    rearranged_tensor_list = []
    # Determine the number of tensors and the number of positions
    num_tensors = len(loaded_tensor_list)

    # Rearrange the tensors
    for i in range(sample_size_per_language+1):
        for j in range(i, num_tensors, sample_size_per_language+1):
            rearranged_tensor_list.append(loaded_tensor_list[j])

    # Print or use the rearranged_tensor_list as needed
    return rearranged_tensor_list


def show_cos_sim(tensor_list, name, calculate = True, save = False, save_path = None, block_size = 100, by_meaning = False, sample_num=100 ):
    # Compute cosine similarity between tensors
    if calculate==True:
        if by_meaning:
            tensor_list = rearrange_by_meaning(tensor_list, sample_num)
        if type(tensor_list)==list:
            tensor_list = np.stack(tensor_list)
        cosine_similarity_matrix = cosine_similarity(tensor_list)
    else:
        cosine_similarity_matrix = tensor_list

    block_matrix = similarity_by_block(cosine_similarity_matrix, block_size)
    diagonal_mean, off_diagonal_mean = compare_similarity_diagnal(block_matrix)


    # Create a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(cosine_similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Cosine Similarity Heatmap of {name}')
    ## write the diagnal mean and off diagnal mean
    plt.xticks(range(len(tensor_list)))
    plt.yticks(range(len(tensor_list)))
    if save:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close()
        # save diagonal_mean, off_diagonal_mean in a csv
        with open(save_path.replace(".png", ".csv"), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Diagonal Mean", "Off Diagonal Mean", "Difference"])
            writer.writerow([diagonal_mean, off_diagonal_mean, diagonal_mean-off_diagonal_mean])


def plot_by_meaning(tensors_file_path, sample_size_per_language):
    with open(tensors_file_path, 'rb') as file:
        loaded_tensor_list = pickle.load(file)
    rearranged_tensor_list = []
    # Determine the number of tensors and the number of positions
    num_tensors = len(loaded_tensor_list)

    # Rearrange the tensors
    for i in range(sample_size_per_language+1):
        for j in range(i, num_tensors, sample_size_per_language+1):
            rearranged_tensor_list.append(loaded_tensor_list[j])

    # Print or use the rearranged_tensor_list as needed
    show_cos_sim(rearranged_tensor_list, "rearranged_by_meaning")



def load_bible_dataset(list_of_lang, sample_num):
    output = []
    for lang in list_of_lang:
        if lang=="en":
            dataset = load_dataset("bible_para", lang1="en", lang2="fr")
            sentences = [render_combined_sentences(i, "en", dataset) for i in range(sample_num)]
            output.append(sentences)
        else:
            langs = ["en", lang]
            langs.sort()
            dataset = load_dataset("bible_para", lang1=langs[0], lang2=langs[1])
            sentences = [render_combined_sentences(i, lang, dataset) for i in range(sample_num)]
            output.append(sentences)
    return output


def load_flores_dataset(list_of_lang, sample_num):
    output = []
    for lang in list_of_lang:
        dataset = load_dataset("Muennighoff/flores200", name=lang)
        sentences = [dataset["dev"][i]["sentence"] for i in range(sample_num)]
        output.append(sentences)
    return output


def find_max_average_neurons(hs_path, mask = (0,500), top_percentage=25, model_layer_num = 24, least=False, random = False, plot = None, return_scores = False):
    
    if type(hs_path) == str:
        with open(hs_path, 'rb') as f:
            hs = pickle.load(f)
        hs = np.array(hs)
    else:
        hs = np.array(hs_path)
    
    if mask is not None:
        hs = hs[mask[0]:mask[1]]

    if random == False:
        # normalize each row
        hs = hs / np.linalg.norm(hs, axis=1, keepdims=True) + 10e-6
        sum_hs = np.zeros(hs.shape[1])
        counter = 0

        for i in tqdm(range(hs.shape[0])):
            for j in range(i+1, hs.shape[0]):
                ## calculate dynamic average of hs
                sum_hs += (hs[i] * hs[j] - sum_hs) / (counter + 1)
                counter += 1
        sum_hs = sum_hs / counter
        counter = 0
        
        if return_scores:
            return sum_hs
    else:
        sum_hs = np.random.rand(hs.shape[1])

    # find the mask of the top values in sum_hs
    neurons_mask = sum_hs >= np.percentile(sum_hs, 100 - top_percentage)

    # find the mask of the least values in sum_hs
    if least:
        neurons_mask = sum_hs <= np.percentile(sum_hs, top_percentage)
    
    # rearrange it according to model structure
    neurons_mask = neurons_mask.reshape(model_layer_num, -1)

    if plot is not None:
        if os.exists(os.path.dirname(plot)):
            os.makedirs(os.path.dirname(plot))

        plt.figure(figsize=(40, 20))
        sns.heatmap(neurons_mask, cmap='Reds')
        plt.xlabel('Neuron', labelpad=15)
        plt.ylabel('Layer', labelpad=28)
        plt.savefig(plot)

    # return a mask of the top values in sum_hs
    return neurons_mask

def zscore_outliers(neuron_scores, threshold = 2, save_path=None):
    ## if threshold is smaller than 1, we calculate percentage instead of sigma
    data = neuron_scores
    mean_performances = np.mean(data, axis=1)
    std_performances = np.std(data, axis=1)

    z_scores = (data - mean_performances[:, np.newaxis]) / std_performances[:, np.newaxis]
    outperformers = z_scores > threshold
    if threshold < 1:
        # for each column, find the top threshold percentage
        top_threshold = np.percentile(z_scores, 100 - threshold * 100, axis=0)
        outperformers = z_scores > top_threshold


    # take the transpose
    outperformers = outperformers.T

    if save_path is not None:
        plt.figure(figsize=(12, 10))
        sns.heatmap(z_scores, cmap='coolwarm', annot=False)
        plt.title('Z-Scores of Performances Across Competitions')
        plt.xlabel('Competition Number')
        plt.ylabel('Participant Index')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    
    return outperformers

class NoiseHookNeuronsBloom:
    def __init__(self, model, neurons_mask, noise_std=0.1, device = "cuda", set_zero = True):
        """
        Initializes the NoiseHook.
        """
        self.model = model
        self.neurons_mask = neurons_mask
        self.noise_std = noise_std
        self._attach_hooks()
        self.device=device
        self.set_zero = set_zero

    def _attach_hooks(self):
        """
        Attaches hooks to the layers specified in the layer_group_to_mask dict.
        """
        self.handles = []  

        for layer in range(len(self.model.transformer.h)):
            
            mask = torch.tensor(self.neurons_mask[layer]).bool().float()

            self.handles.append(self.model.transformer.h[layer].mlp.gelu_impl.register_forward_hook(self._make_hook(mask)))

    def _make_hook(self, mask):
        """
        Creates a hook function with the specified mask.

        Args:
        - mask (torch.Tensor): The mask specifying where to add noise.

        Returns:
        - A hook function that adds noise according to the specified mask.
        """
        def hook(module, input, output):

            if self.set_zero:
                return output * (1-mask.to(output.get_device()))
            else : 
                noise = torch.randn_like(output) * self.noise_std
                noise = noise * mask
                return output + noise
        
        return hook
    

    def remove_hooks(self):
        """
        Removes all hooks from the model.
        """
        for handle in self.handles:
            handle.remove()

class NoiseHookNeuronsLlama:
    def __init__(self, model, neurons_mask, noise_std=0.1, device = "cuda", set_zero = True):
        """
        Initializes the NoiseHook.
        """
        self.model = model
        self.neurons_mask = neurons_mask
        self.noise_std = noise_std
        self._attach_hooks()
        self.device=device
        self.set_zero = set_zero

    def _attach_hooks(self):
        """
        Attaches hooks to the layers specified in the layer_group_to_mask dict.
        """
        self.handles = []  

        for layer in range(len(self.model.model.layers)):
            
            mask = torch.tensor(self.neurons_mask[layer]).bool().float()

            self.handles.append(self.model.model.layers[layer].mlp.act_fn.register_forward_hook(self._make_hook(mask)))

    def _make_hook(self, mask):
        """
        Creates a hook function with the specified mask.

        Args:
        - mask (torch.Tensor): The mask specifying where to add noise.

        Returns:
        - A hook function that adds noise according to the specified mask.
        """
        def hook(module, input, output):

            if self.set_zero:
                return output * (1-mask.to(output.get_device()))
            else : 
                noise = torch.randn_like(output) * self.noise_std
                noise = noise * mask
                return output + noise
        
        return hook
    

    def remove_hooks(self):
        """
        Removes all hooks from the model.
        """
        for handle in self.handles:
            handle.remove()
            
def eval_languages_ppl_on_xlsum(model, tokenizer, device, languages = ["english", "chinese_simplified", "french", "spanish", "portuguese", "arabic", "vietnamese", "hindi", "indonesian"], save_path = None, bs=1):

    for language in languages:
        ppl = eval_ppl_multilingual(model, tokenizer, device, language = language, bs=bs)
        print(f"ppl {language} = {ppl}")
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if not os.path.exists(save_path):
                with open(save_path, 'a') as f:
                    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["language", "ppl"])
            with open(save_path, 'a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([language, ppl])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--dataset_name", type=str, default="bible")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str)
    parser.add_argument("--deactivate", type=str, default=False)
    parser.add_argument("--evaluate_ppl", type=bool, default=False)
    parser.add_argument("--evaluate_tasks", type=bool, default=False)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--random_percentage", type=int, default=0)

    args = parser.parse_args()
    time_now = time.strftime("%Y-%m-%d-%X")

    sample_num = args.sample_num
    device = args.device
    dataset_name = args.dataset_name


    sentences_list = []

    threshold = args.threshold
    random_percentage = args.random_percentage
    deactivate = args.deactivate
    tokenize = True
    model_name_or_path = args.model
    
    
    print("Loading model...")

    if args.revision is not None:
        if "bloom" in model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", use_fast=True,  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=args.revision, device_map="balanced", torch_dtype="auto",  trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True,  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",  trust_remote_code=True)


    # Here we choose the hidden state number according to the model name. You can change it according to your model.
    if "llama" in model_name_or_path or "baichuan" in model_name_or_path:
        bs=1
        if "Llama-3" in model_name_or_path:     
            model.seqlen = 8192
            hidden_state_num = model.model.layers[0].mlp.up_proj.out_features
        elif "Llama-2" in model_name_or_path:
            model.seqlen = 4096
            hidden_state_num = model.model.layers[0].mlp.up_proj.out_features
            print(hidden_state_num)
        else:
            model.seqlen = 2048
            hidden_state_num = 11008
    elif "bloom" in model_name_or_path:
        bs=7
        model.seqlen = 2048
        hidden_state_num = model.transformer.h[0].mlp.dense_h_to_4h.out_features
    

    


    print("#--------------------Loading dataset...Preparing samples...-----------------------------")
    
    if args.dataset_name=="bible":
        dataset_list = ["en", "zh", "fr", "es", "pt", "ar", "vi", "hi", "id"]
        sentences_list = load_bible_dataset(dataset_list, sample_num)
        tokenize = True
    elif args.dataset_name=="flores":
        dataset_list = ["eng_Latn", "zho_Hans", "zho_Hant", "fra_Latn", "spa_Latn", "por_Latn", 
            "arb_Arab", "vie_Latn", "hin_Deva", "ind_Latn" ]

        sentences_list = load_flores_dataset(dataset_list, sample_num)
        tokenize = True
    else:
        raise("Dataset not supported, consider modify the load_xxx_dataset function.")



    print("#--------------------LRDS and SADS calculation-----------------------------")

    outputs = []

    if "llama" in model_name_or_path:
        model_layer_num = len(model.model.layers)
        for sentences in tqdm(sentences_list):
            outputs += generate_hidden_states_llama(sentences, model, device, tokenizer, tokenize = tokenize, hs_dim = hidden_state_num)
    elif "bloom" in model_name_or_path:
        model_layer_num = len(model.transformer.h)
        for sentences in tqdm(sentences_list):
            outputs += generate_hidden_states_bloom(sentences, model, device, tokenizer, tokenize = tokenize, hs_dim = hidden_state_num)
    else:
        raise("Model not recognized")
    

    print("Calculating cosine similarity...")

    outputs = np.array(outputs)
    ## key linguistic region development scores 
    print("Calculating Key Linguisitic Region Development Scores")
    show_cos_sim(outputs, f"{args.model.split('/')[-1]}_hidden_states", save = True, save_path = f"./results_graph_{args.model.split('/')[-1]}/{args.revision}/hidden_states_{time_now}_{args.dataset_name}_bylanguage.png", block_size = sample_num)
    ## Semantic alignment development scores
    print("Calculating Semantic Alignment Development Scores")
    show_cos_sim(outputs, f"{args.model.split('/')[-1]}_hidden_states", save = True, save_path = f"./results_graph_{args.model.split('/')[-1]}/{args.revision}/hidden_states_{time_now}_{args.dataset_name}_bymeaning.png", by_meaning = True, block_size = len(dataset_list), sample_num=sample_num)
    
    ## Scores by layer：
    for i in range(model_layer_num):
        print(f"Layer {i}")
        layer_output = outputs[:, i*hidden_state_num:(i+1)*hidden_state_num]
        ## key linguistic region development scores 
        print("Calculating Key Linguisitic Region Development Scores")
        show_cos_sim(layer_output, f"{args.model.split('/')[-1]}_hidden_states_{i}", save = True, save_path = f"./results_graph_{args.model.split('/')[-1]}/{args.revision}/hidden_states_{i}_{time_now}_{args.dataset_name}_bylanguage.png", block_size = sample_num)
        ## Semantic alignment development scores
        print("Calculating Semantic Alignment Development Scores")
        show_cos_sim(layer_output, f"{args.model.split('/')[-1]}_hidden_states_{i}", save = True, save_path = f"./results_graph_{args.model.split('/')[-1]}/{args.revision}/hidden_states_{i}_{time_now}_{args.dataset_name}_bymeaning.png", by_meaning = True, block_size = len(dataset_list))



    print("#--------------------Important Neurons Probing-----------------------------")

    neurons_scores = []
    for i in range(0, len(dataset_list)):
        neurons_score = find_max_average_neurons(hs_path = outputs, mask = (i*sample_num, (i+1)*sample_num), model_layer_num = model_layer_num, plot = None, return_scores=True)
        neurons_scores.append(neurons_score)

    neurons_scores = np.stack(neurons_scores, axis=-1)
    outliers = zscore_outliers(neurons_scores, threshold = threshold)


    ## The key neuron number of each language in each layer is printed here, showing the change of language key region size with different layers.
    if "bloom" in model_name_or_path:
        x_reshaped = outliers.reshape(len(dataset_list), model_layer_num, model.transformer.h[0].mlp.dense_h_to_4h.out_features)
    elif "Llama" in model_name_or_path:
        x_reshaped = outliers.reshape(len(dataset_list), model_layer_num, model.model.layers[0].mlp.up_proj.out_features)
    result = x_reshaped.sum(axis=2)
    result_list = result.tolist()
    save_path = f'./outputs/{args.model.split("/")[-1]}_{args.revision}/outliers_{sample_num}_{time_now}_{args.dataset_name.replace("/", "-")}.csv'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in result_list:
            writer.writerow(row)
    
    print("The key neuron number of each language in each layer is saved in ", save_path)

    
    ## If we want to randomly select some neurons as a baseline, we can use the following code. Note that we need to calculate the number of important neurons we found, and then modify random_percentage in the following code to keep the changes on both sides consistent.
    if random_percentage!=0:
        outliers=[]
        outliers.append(find_max_average_neurons(hs_path = outputs, mask = (0, sample_num), top_percentage=random_percentage, model_layer_num = model_layer_num, plot = None, random = True).flatten())
        outliers = np.array(outliers)

    print(f"Number of important neurons respectively in {dataset_list} are:")
    print(outliers.sum(axis=1))



    print("--------------------Deactivation and perplecity evaluation-----------------------------")

    counter=0

    for neurons_mask in outliers:

        if deactivate:
            neurons_mask = neurons_mask.reshape(model_layer_num, -1)
            ## print sum for each layer
            if "bloom" in model_name_or_path:
                noise_hook = NoiseHookNeuronsBloom(model, neurons_mask, device = device, set_zero = True)
            elif "llama" in model_name_or_path:
                bs=1
                noise_hook = NoiseHookNeuronsLlama(model, neurons_mask, device = device, set_zero = True)
            else:
                raise("Model non recognized")
        else:
            threshold = "nan"
            dataset_list = ["nan"]
            
        if args.evaluate_ppl:
            ## For each language, we deactivate the important neurons and test the ppl. When testing this, deactivate should be set to True, and the important neurons will be zeroed out.

            eval_languages_ppl_on_xlsum(model, tokenizer, device,
                                        languages = [
                                                    "english", "chinese_simplified", 
                                                    "french", "spanish", "portuguese", 
                                                    "chinese_traditional", 
                                                    "arabic", "vietnamese", "hindi", "indonesian", 
                                                    ],
                                        save_path = f"./results_{model_name_or_path.split('/')[-1]}_{args.revision}/remove_outlier_threshold_{threshold}_neurons_{time_now}/ppl_removed_{dataset_list[counter]}.csv", bs=bs)
        if deactivate:
            noise_hook.remove_hooks()

        counter+=1


    print("--------------------Cross-lingual 0-shot task-----------------------------")

    if args.evaluate_tasks:
        task_names = ["xstorycloze"]

        print("Evaluating Cross-lingual 0-shot task on xstorycloze...")
        
        results = lm_eval.evaluator.simple_evaluate(
            model=lm_eval.models.huggingface.HFLM(pretrained = model, tokenizer = tokenizer, device = device),
            tasks=task_names,
            batch_size="auto",   
            device=device,
            log_samples = False
        )

        dumped = json.dumps(results, indent=2)
        print(dumped)

        dirname = f"./results_{model_name_or_path.split('/')[-1]}_{args.revision}/remove_outlier_threshold_{threshold}_neurons_{time_now}/tasks_removed_{dataset_list[counter].split('/')[-1]}.json"
        if not os.path.exists(os.path.dirname(dirname)):
            os.makedirs(os.path.dirname(dirname))
        with open(dirname, "w") as f:
            f.write(dumped)
    



