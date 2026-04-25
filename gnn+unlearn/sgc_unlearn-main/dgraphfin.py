from typing import Optional, Callable, List
import os
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


def _index_to_mask(index: torch.Tensor, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index.long()] = True
    return mask


def read_dgraphfin(folder):
    print('read_dgraphfin')
    names = ['dgraphfin.npz']
    items = [np.load(folder+'/'+name) for name in names]
    
    x = items[0]['x']
    y = items[0]['y'].reshape(-1,1)
    edge_index = items[0]['edge_index']
    edge_type = items[0]['edge_type']
    train_mask = items[0]['train_mask']
    valid_mask = items[0]['valid_mask']
    test_mask = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.float)
    train_index = torch.tensor(train_mask, dtype=torch.int64)
    valid_index = torch.tensor(valid_mask, dtype=torch.int64)
    test_index = torch.tensor(test_mask, dtype=torch.int64)

    train_mask = _index_to_mask(train_index, x.size(0))
    valid_mask = _index_to_mask(valid_index, x.size(0))
    test_mask = _index_to_mask(test_index, x.size(0))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask
    data.train_idx = train_index
    data.valid_idx = valid_index
    data.test_idx = test_index

    return data

class DGraphFin(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"dgraphfin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ''

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        root = self._resolve_root(root, name)
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @staticmethod
    def _resolve_root(root: str, name: str) -> str:
        dataset_name = name
        raw_file = 'dgraphfin.npz'

        script_dir = osp.dirname(osp.abspath(__file__))
        project_root = osp.abspath(osp.join(script_dir, '..', '..'))
        workspace_root = osp.abspath(osp.join(project_root, '..'))
        abs_root = osp.abspath(root)

        candidates = []
        for candidate in [
            abs_root,
            osp.dirname(abs_root),
            osp.join(workspace_root, 'PyG_datasets'),
            osp.join(project_root, 'PyG_datasets'),
            osp.join(script_dir, 'PyG_datasets'),
            workspace_root,
            project_root,
            script_dir,
        ]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            # candidate is already dataset dir, e.g. .../DGraphFin
            if osp.basename(candidate).lower() == dataset_name.lower():
                if osp.exists(osp.join(candidate, 'raw', raw_file)):
                    resolved = osp.dirname(candidate)
                    if resolved != abs_root:
                        print(f"[DGraphFin] Auto-resolved root: {resolved}")
                    return resolved
            # candidate is root dir, e.g. .../PyG_datasets
            if osp.exists(osp.join(candidate, dataset_name, 'raw', raw_file)):
                if candidate != abs_root:
                    print(f"[DGraphFin] Auto-resolved root: {candidate}")
                return candidate

        return root

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['dgraphfin.npz']
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
#         for name in self.raw_file_names:
#             download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_dgraphfin(self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
