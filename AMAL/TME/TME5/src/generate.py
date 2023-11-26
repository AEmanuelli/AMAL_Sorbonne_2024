from textloader import *
import math
import torch
from utils import *

#  TODO:  Ce fichier contient les différentes fonction de génération 


def generate(model, embedding, eos=1, start_string="Trump", generation_length=100, temperature=0.5):
    input_eval = string2code(start_string).to(device)  # Convert starting string to tensor
    input_eval = input_eval.unsqueeze(0)  # Add batch dimension

    generated_text = start_string

    model.eval()  # Evaluation mode
    with torch.no_grad():
        # Initialize hidden and cell states for LSTM
        if isinstance(model, LSTMModel):
            hidden = torch.zeros(model.num_layers, 1, model.latent_dim).to(device)
            cell = torch.zeros(model.num_layers, 1, model.latent_dim).to(device)
            hidden_state = (hidden, cell)
            for i in range(generation_length):
                # Get the last character, embed, and add an extra dimension for batch
                last_char = embedding(input_eval[:, -1]).unsqueeze(0).float()
            
                # Forward pass through the model
                output, hidden_state = model(last_char, hidden_state)
                output = output.view(-1, model.output_dim)
                    # Apply temperature sampling to the output logits
                next_char_idx = temperature_sampling(output, temperature)
                if next_char_idx == eos:
                    break
                next_char_idx = next_char_idx.squeeze().tolist()

                # Convert to string and append
                next_char = code2string([next_char_idx])
                generated_text += next_char

                # Update input for next generation step
                next_char_tensor = torch.tensor([[next_char_idx]], device=device)
                input_eval = torch.cat((input_eval, next_char_tensor), dim=1)
        elif isinstance(model, (RNN)):
            # For RNN and GRU, hidden state only
            h = torch.zeros([1, model.latent_dim], device=input_eval.device)  # Initialize hidden state
            for i in range(generation_length):
                # Update hidden state
                h = model.one_step(embedding(input_eval[:, -1]).float(), h)
                logits = model.decode(h)
                next_char_idx = temperature_sampling(logits, temperature)
                if next_char_idx==eos:
                    break
                next_char_idx = next_char_idx.squeeze().tolist()  # Convert to list

                if isinstance(next_char_idx, int):
                    next_char_idx = [next_char_idx]

                next_char = code2string(next_char_idx)

                generated_text += next_char

                # Update input for next generation step
                next_char_tensor = torch.tensor([next_char_idx], device=input_eval.device)
                input_eval = torch.cat((input_eval, next_char_tensor), dim=1)
        elif isinstance(model, (GRUModel)):
                current_batch_size = 1  # For generation, batch size is typically 1

                h = model.init_hidden(current_batch_size)
                
                for i in range(generation_length):
                    # Update hidden state
                    embedded = embedding(input_eval[:, -1]).float().unsqueeze(0)  # Add batch dimension
                    output, h = model(embedded, h)
                    logits = output.squeeze(0)  # Remove batch dimension for sampling

                    next_char_idx = temperature_sampling(logits, temperature)
                    if next_char_idx == eos:
                        break

                    next_char_idx = next_char_idx.item()  # Get the actual index value
                    next_char = code2string([next_char_idx])

                    generated_text += next_char

                    # Update input for next generation step
                    next_char_tensor = torch.tensor([[next_char_idx]], device=input_eval.device)
                    input_eval = torch.cat((input_eval, next_char_tensor), dim=1)
        else:
            raise ValueError("Unsupported model type.")

    return generated_text




def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
    Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
    * rnn : le réseau
    * emb : la couche d'embedding
    * decoder : le décodeur
    * eos : ID du token end of sequence
    * k : le paramètre du beam-search
    * start : début de la phrase
    * maxlen : longueur maximale
    """
    # Initial setup
    beams = [(start, 0)]  # Each beam is a tuple of (sequence, log_probability)
    for _ in range(maxlen):
        candidates = []
        for seq, score in beams:
            input_tensor = torch.tensor([string2code(seq[-1])]) if seq else torch.tensor([0])
            embedded = emb(input_tensor)
            output, hidden = rnn(embedded)
            logits = decoder(output)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Consider the top k candidates for each beam
            topk_probs, topk_indices = torch.topk(log_probs, k)
            for i in range(k):
                next_char = id2lettre(topk_indices[0][i].item())
                new_seq = seq + next_char
                new_score = score + topk_probs[0][i].item()
                candidates.append((new_seq, new_score))

        # Select the top k beams
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
        if all(seq[-1] == eos for seq, _ in beams):
            break

    return max(beams, key=lambda x: x[1])[0]  # Return the sequence with the highest score

# Implementing the p_nucleus function
def p_nucleus(decoder, alpha):
    """
    Creates a function for nucleus sampling.
    """
    def compute(h):
        logits = decoder(h)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        idx = (cumulative_probs > alpha).nonzero()[0]
        effective_logits = sorted_logits[:, :idx+1]
        probabilities = torch.nn.functional.softmax(effective_logits, dim=-1)
        next_char_idx = torch.multinomial(probabilities, 1)
        return sorted_indices[:, :idx+1][0, next_char_idx]
    
    return compute
