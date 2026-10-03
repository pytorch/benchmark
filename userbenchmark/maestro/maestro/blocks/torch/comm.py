from abc import abstractmethod
import torch.distributed as dist
from typing import Optional

from core.block import CommBlock
from core.axis import TorchAxis, current_axis
from core.utils.logging import get_logger


logger = get_logger(__name__)


class _TorchCommBlock(CommBlock):
    """Communication block"""
    collective_fnc = None

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def run(self):
        if self.__class__.collective_fnc is None:
            raise ValueError(f"collective_fnc attribute is not defined in {self.__class__.__name__}")

        with self.axis.use_axis():
            self.__class__.collective_fnc(self.dst_buf, self.src_buf, group=self.axis.my_group)
    

class TorchAllToAll(_TorchCommBlock):
    collective_fnc = dist.all_to_all_single
    registry_name = "torch_alltoall"

    @staticmethod
    def get_bus_bw_factor(team_size: int):
        return (team_size-1)/team_size


class TorchReduceScatter(_TorchCommBlock):
    collective_fnc = dist.reduce_scatter_tensor
    registry_name = "torch_reduce_scatter"

    @classmethod
    def get_dst_buf_size(cls, full_vector_size, team_size):
        return full_vector_size // team_size

    @staticmethod
    def get_bus_bw_factor(team_size: int):
        return (team_size-1)/team_size
    

class TorchAllGather(_TorchCommBlock):
    collective_fnc = dist.all_gather_into_tensor
    registry_name = "torch_all_gather"

    def _get_full_vector_count(self):
        return self.dst_buf.shape[0]
    
    @classmethod
    def get_src_buf_size(cls, full_vector_size, team_size):
        if full_vector_size % team_size != 0:
            raise ValueError(f"Full vector size must be divisible by team_size={team_size} for {cls.__name__}")
        return full_vector_size // team_size

    @staticmethod
    def get_bus_bw_factor(team_size: int):
        return (team_size-1)/team_size


class TorchReduce(_TorchCommBlock):
    registry_name = "torch_reduce"

    @classmethod
    def collective_fnc(cls, dst_buf, src_buf, group):
        dist.reduce(dst_buf, dst=current_axis.get().root_rank, group=group)

    @classmethod
    def get_src_buf_size(cls, full_vector_size, team_size):
        return 0


class TorchSendRecvRing(_TorchCommBlock):
    registry_name = "torch_send_recv_ring"

    @staticmethod
    def collective_fnc(dst, src, group):
        ranks = dist.get_process_group_ranks(group)
        my_index = ranks.index(dist.get_rank())
        send_index = (my_index + 1) % len(ranks)
        recv_index = (my_index - 1) % len(ranks)
        send_rank = ranks[send_index]
        recv_rank = ranks[recv_index]
        logger.debug(f"[{dist.get_rank()}] SendRecvRing: sendto={send_rank}, recvfrom={recv_rank}, ranks={ranks}, {my_index=}, {send_index=}, {recv_index=}")
        send_op = dist.P2POp(dist.isend, src, send_rank, group=group)
        recv_op = dist.P2POp(dist.irecv, dst, recv_rank, group=group)
        return dist.batch_isend_irecv([send_op, recv_op])


class TorchAllReduce(_TorchCommBlock):
    registry_name = "torch_all_reduce"
    
    @classmethod
    def collective_fnc(cls, dst_buf, src_buf, group):
        dist.all_reduce(dst_buf, group=current_axis.get().my_group)
    
    def get_src_buf_size(self, *args, **kwargs):
        # allreduce uses src buffer only
        return 0
    
    @staticmethod
    def get_bus_bw_factor(team_size: int):
        return 2*(team_size-1)/team_size


