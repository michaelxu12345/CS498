###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, world, rank, iter):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    
    send_to = (rank+1) % world
    rcv_from = (rank - 1 + world) % world
    
    send_idx = (rank - iter + world) % world
    rcv_idx = (rank - iter + 2* world - 1) % world
    
    to_send = chunks[send_idx].clone()
    s = dist.isend(to_send, dst = send_to)
    
    rcv_buf = chunks[rcv_idx].clone()
    r = dist.irecv(rcv_buf, src=rcv_from)
    
    s.wait()
    r.wait()
    chunks[rcv_idx] += rcv_buf
    
    return chunks
        
def all_gather(chunks, world, rank, iter):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    send_to = (rank+1) % world
    rcv_from = (rank - 1 + world) % world
    
    send_idx = (rank - iter + world + 1) % world
    rcv_idx = (rank - iter + 2* world) % world
    
    to_send = chunks[send_idx].clone()
    s = dist.isend(to_send, dst = send_to)
    
    rcv_buf = chunks[rcv_idx].clone()
    r = dist.irecv(rcv_buf, src=rcv_from)
    
    s.wait()
    r.wait()
    chunks[rcv_idx] = rcv_buf
    
    return chunks

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padded_flat = torch.nn.functional.pad(flat, (0, n - chunk * world), 'constant', 0)
    # modify this line and fill correct value into padded_flat
    
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    for i in range(world - 1):
        reduce_scatter(chunks, world, rank, i)
    
    for i in range(world - 1):
        all_gather(chunks, world, rank, i)
    
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    
    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return