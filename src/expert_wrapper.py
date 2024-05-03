import typing as tp
import torch
import torch.nn as nn

from utils import nested_flatten, nested_pack

class SwitchExpertWrapper(nn.Module):
    def __init__(
        self,
        expert_module: tp.Any,
        device: torch.device,
    ):
        super().__init__()
        
        self.expert_module, self.storage = self.replace_layer_storage(expert_module, device)
        # self.expert_module = lambda *args, **kwargs: expert_module(*args, **kwargs)
        
        self._register_state_dict_hook(self._add_storage_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_storage_from_state_dict_hook)

    @staticmethod
    def _add_storage_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'storage'] = torch.as_tensor(self.storage, dtype=torch.bfloat16)
        return state_dict

    def _load_storage_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.storage.copy_(state_dict[prefix + 'storage'].storage().untyped())
        del state_dict[prefix + 'storage']

    def forward(self, *args, **kwargs):
        return self.expert_module(*args, **kwargs)

    @staticmethod
    def replace_layer_storage(
        layer: tp.Any,
        device: torch.device,
    ):
        '''
        这个静态方法的目的是将传入的 layer（一个包含多个子层的模块）的所有张量转移到一个单一的 UntypedStorage 对象中。这样做的优点包括：
        - 内存连续性：将所有小张量合并到一个大的连续内存块中，有助于减少内存碎片和提高缓存效率。
        - 优化数据传输：在多设备环境下，如数据需要在 CPU 和 GPU 之间移动，使用单一的存储可以减少同步和数据传输的开销。
        - 减少内存占用：相较于分散存储，连续存储往往可以更好地利用内存，减少总体占用空间。
        '''
        state_dict = {
            f"w{i}": {
                "weight": getattr(layer, f"w{i}").weight,
            }
            for i in ['i', 'o']
        }

        storage_size = 0
        offsets = [0]

        for x in nested_flatten(state_dict):
            if not isinstance(x, torch.Tensor):
                continue
            storage_size += x.nbytes
            offsets.append(storage_size)

        storage = torch.UntypedStorage(storage_size, device=device) 

        i = 0
        new_flattened_states = list()
        for x in nested_flatten(state_dict):
            if not isinstance(x, torch.Tensor):
                new_flattened_states.append(x)
                continue

            start = offsets[i]
            end = offsets[i + 1]
            a_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device=device).view(x.shape)
            a_view[...] = x
            assert a_view.data_ptr() == storage.data_ptr() + start
            i += 1
            new_flattened_states.append(a_view)

        state_dict = nested_pack(new_flattened_states, state_dict)

        for layer_id, states in state_dict.items():
            patched = getattr(layer, layer_id)
            patched.weight = nn.Parameter(states['weight'])
            setattr(layer, layer_id, patched)

        return layer, storage