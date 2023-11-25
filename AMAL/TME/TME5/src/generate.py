from textloader import string2code, id2lettre
import math
import torch
from tp5 import *

#  TODO:  Ce fichier contient les différentes fonction de génération


def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """
    Generates a sequence using an RNN. The sequence starts with 'start' (or empty if start is not provided) and continues until the 'eos' token is generated or maxlen is reached.
    
    Args:
    * rnn (nn.Module): The RNN model.
    * emb (function): Embedding layer function.
    * decoder (function): Decoder function that returns the logits of possible outputs.
    * eos (int): ID of the end-of-sequence token.
    * start (str): Starting string for the sequence.
    * maxlen (int): Maximum length of the generated sequence.

    Returns:
    * str: The generated sequence.
    """
    sequence = start
    hidden = None
    str = start

    for _ in range(maxlen):
        input_tensor = torch.tensor([string2code(sequence[-1])]) if sequence else torch.tensor([0])
        embedded = emb(input_tensor)
        output, hidden = rnn(embedded, hidden) if hidden is not None else rnn(embedded)
        logits = decoder(output)
        next_char_idx = temperature_sampling(logits, temperature)
        next_char_idx = next_char_idx.squeeze().tolist()  # Convert to list

        # Check if it's a single integer, convert to list if so
        if isinstance(next_char_idx, int):
            next_char_idx = [next_char_idx]

        next_char = id2lettre(next_char_idx)

        str += next_char

        if next_char_idx == eos:
            break

    return str

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
