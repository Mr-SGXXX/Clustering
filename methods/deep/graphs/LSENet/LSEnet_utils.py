import torch
import torch.nn.functional as F

def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    adjacency = torch.zeros(N, N).to(edge_index.device)
    m = edge_index.shape[1]
    if weight is None:
        adjacency[edge_index[0], edge_index[1]] = 1
    else:
        adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    if is_sparse:
        weight = weight if weight is not None else torch.ones(m).to(edge_index.device)
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    return adjacency

def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation is None:
        return None
    else:
        raise NotImplementedError('the non_linear_function is not implemented')
    
    
def gumbel_sigmoid(logits, tau: float = 1, hard: bool = False, threshold: float = 0.5):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret