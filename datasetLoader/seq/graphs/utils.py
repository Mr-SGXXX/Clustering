from torch_sparse import SparseTensor
import torch

def count_edges(sparse_adj: SparseTensor) -> int:
    """
    Counts the number of edges in an undirected graph given its adjacency matrix.
    """
    # make sure the input graph is undirected
    # if not sparse_adj.is_symmetric():
    #     raise ValueError("Input graph must be undirected (sym) for this function.")

    # get the row and column indices of the adjacency matrix
    row = sparse_adj.storage.row()
    col = sparse_adj.storage.col()

    # count the number of non-zero elements in the adjacency matrix
    num_nonzeros = sparse_adj.nnz()

    # count the number of self-loops
    num_selfloops = torch.sum(row == col).item()

    # count the number of edges in the graph
    num_edges = (num_nonzeros - num_selfloops) // 2

    return num_edges