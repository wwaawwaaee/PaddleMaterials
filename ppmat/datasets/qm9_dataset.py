# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import annotations

import math
import os
import os.path as osp
import pickle ## for dump/load
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.io import read_json
from ppmat.utils.misc import is_equal

try:
    from tqdm import tqdm 
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class QM9Dataset(Dataset):
    """
    QM9 (GDB-9) Dataset Handler
    ——by wwaawwaaee
    
    **Dataset Overview**
    this class downloads QM9 primiry(end with .xyz)and transfered into 
    the input structures and quantum chemical property labels required 
    by graph neural network (GNN) models.(maybe)

    **dataset format**
    -----------------
    - raw data: qm9.zip ()
    - struncture file: a sample corresponds to a seperate .xyz file
    - attribute(label): 19 quantum chemical properties are embedded in each .xyz file's
    second line comment

    **source**:Original data available at https://figshare.com/ndownloader/files/3195389

    The dataset can also be found at https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/dsgdb9nsd.xyz.tar.bz2
    
   **Key Properties List (Available for 'property_names' argument)**
    -----------------------------------------------------------------
    1.  mu (Dipole Moment, Debye)
    2.  alpha (Isotropic Polarizability, Bohr^3)
    3.  homo (HOMO Energy, Hartree)
    4.  lumo (LUMO Energy, Hartree)
    5.  gap (LUMO-HOMO Gap, Hartree)
    6.  U0 (Internal Energy at 0 K, Hartree)
    # ... (remaining 13 properties here in their correct order)
    
    **__getitem__ Sample Contract**
    ----------------------------------------
    - 'atom_types': np.ndarray (dtype=int64) - Atomic numbers (Z).
    - 'coords': np.ndarray (dtype=float32) - 3D Cartesian coordinates in Angstrom.
    - [property_name]: np.ndarray (dtype=float32) - The target label value (e.g., 'lumo').
    - 'graph': (Optional) The graph object constructed by the converter (if configured).
    
    Args:
        path (str): The root directory to store downloaded and cache files.
        property_names (Union[str, List[str]]): The name(s) of the target property 
            to predict. Must be selected from the list above. Defaults to 'lumo'.
        build_graph_cfg (Dict, optional): Configuration dictionary for building 
            the graph representation from the molecular structure (e.g., cutoff radius). 
            Defaults to None (structure is returned instead of graph).
        transforms (Optional[Callable], optional): A preprocessing function to apply 
            to the sample dictionary. Defaults to None.
        cache_path (Optional[str], optional): Explicit path for the cache directory. 
            Defaults to None.
        overwrite (bool, optional): If True, forces the rebuilding of caches. 
            Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out corrupted samples. 
            Defaults to True.
    """

    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/dsgdb9nsd.xyz.tar.bz2"
    name = "qm9"
    md5 = "AD1EBD51EE7F5B3A6E32E974E5D54012"

def __init__(
        self,
        path: str,
        property_names: Union[str, List[str]] = "lumo",
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        # 1. 初始化参数
        self.root = path
        # 压缩包下载后的本地路径
        self.download_path = osp.join(path, osp.basename(self.url))
        # 解压后的核心数据文件路径 (假设解压后是 dsgdb9nsd.xyz)
        # 注意：tar.bz2 解压后文件名通常去掉后缀，具体取决于压缩包结构
        # 这里假设解压后得到一个合并好的 xyz 文件
        self.raw_xyz_path = osp.join(path, "dsgdb9nsd.xyz")

        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = property_names
        
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms
        
        # 2. 设置缓存路径
        if cache_path is None:
            # 根据图转换配置生成唯一的缓存目录名
            if build_graph_cfg is not None:
                graph_name = build_graph_cfg.get("__class_name__", "custom")
                cutoff = build_graph_cfg.get("__init_params__", {}).get("cutoff", "default")
                suffix = f"_{graph_name}_cutoff_{cutoff}"
            else:
                suffix = "_no_graph"
            cache_path = osp.join(path, f"qm9_cache{suffix}")
        
        self.cache_path = cache_path
        os.makedirs(self.cache_path, exist_ok=True)
        logger.info(f"Cache directory: {self.cache_path}")

        # 3. 核心逻辑：检查缓存 -> 下载处理 -> 加载数据
        self._prepare_data(overwrite)

        # 4. 加载到内存 (Properties & Structures/Graphs paths)
        self._load_data_references()

        # 5. 过滤无效数据
        if filter_unvalid:
            self._filter_unvalid()

    def _prepare_data(self, overwrite: bool):
        """准备数据：下载、解压、解析、构建图并缓存"""
        prop_cache_dir = osp.join(self.cache_path, "properties")
        struct_cache_dir = osp.join(self.cache_path, "structures")
        graph_cache_dir = osp.join(self.cache_path, "graphs")
        
        # 检查是否需要重新处理基础数据 (结构和属性)
        need_process_raw = overwrite or not (osp.exists(prop_cache_dir) and osp.exists(struct_cache_dir))
        
        if need_process_raw:
            # A. 下载与解压
            if not osp.exists(self.raw_xyz_path):
                self._download_and_extract()
            
            # B. 解析原始 XYZ
            if dist.get_rank() == 0:
                logger.info("Processing raw QM9 data (parsing XYZ)...")
                os.makedirs(prop_cache_dir, exist_ok=True)
                os.makedirs(struct_cache_dir, exist_ok=True)
                
                structures, properties = self._process_raw_xyz()
                
                # C. 写入缓存 (Structure & Properties)
                num_samples = len(structures)
                logger.info(f"Saving {num_samples} structures and properties to cache...")
                
                # 保存结构 (分文件保存以便随机读取)
                for i, struct in enumerate(structures):
                    self._save_cache(osp.join(struct_cache_dir, f"{i:06d}.pkl"), struct)
                
                # 保存属性 (按列保存)
                for key, vals in properties.items():
                    self._save_cache(osp.join(prop_cache_dir, f"{key}.pkl"), vals)
                
                # 保存配置元数据
                self._save_cache(osp.join(self.cache_path, "meta.pkl"), {"num_samples": num_samples})
            
            if dist.is_initialized():
                dist.barrier()

        # 检查是否需要构建图
        if self.build_graph_cfg is not None:
            # 加载元数据获取样本数
            meta = self._load_cache(osp.join(self.cache_path, "meta.pkl"))
            num_samples = meta["num_samples"]
            
            # 简单判断：如果图缓存数量不对，则重新构建
            need_build_graph = overwrite or not osp.exists(graph_cache_dir) or \
                               (len(os.listdir(graph_cache_dir)) != num_samples)
            
            if need_build_graph:
                if dist.get_rank() == 0:
                    logger.info("Building graphs from structures...")
                    os.makedirs(graph_cache_dir, exist_ok=True)
                    converter = build_graph_converter(self.build_graph_cfg)
                    
                    # 批量处理以节省内存
                    batch_size = 5000
                    for start_idx in range(0, num_samples, batch_size):
                        end_idx = min(start_idx + batch_size, num_samples)
                        # 加载结构
                        batch_structs = [
                            self._load_cache(osp.join(struct_cache_dir, f"{i:06d}.pkl")) 
                            for i in range(start_idx, end_idx)
                        ]
                        # 转换
                        batch_graphs = converter(batch_structs)
                        # 保存
                        for i, g in enumerate(batch_graphs):
                            idx = start_idx + i
                            self._save_cache(osp.join(graph_cache_dir, f"{idx:06d}.pkl"), g)
                        logger.info(f"Built graphs {end_idx}/{num_samples}")
                
                if dist.is_initialized():
                    dist.barrier()

    def _process_raw_xyz(self):
        """解析 dsgdb9nsd.xyz 文件"""
        # 使用 ASE 读取所有帧
        logger.info(f"Reading {self.raw_xyz_path} via ASE...")
        atoms_list = ase.io.read(self.raw_xyz_path, index=':')
        
        # 读取文件文本用于解析属性 (因为ASE解析这部分可能不统一)
        with open(self.raw_xyz_path, 'r') as f:
            lines = f.readlines()
            
        structures = []
        properties = defaultdict(list)
        
        # 简单的状态机解析 XYZ
        current_line = 0
        idx = 0
        
        while current_line < len(lines) and idx < len(atoms_list):
            try:
                # Line 1: 原子数
                num_atoms = int(lines[current_line].strip())
                # Line 2: 属性行 (tag, index, A, B, C, mu, ...)
                prop_line = lines[current_line + 1]
                # 去掉 'gdb' 这种非数字字符
                raw_props = prop_line.replace('gdb', '').strip().split()
                # 转换为浮点数
                float_props = [float(x) for x in raw_props]
                
                # 存储属性
                # 注意：raw_props[0] 是 index, raw_props[1] 是 A ...
                # 我们按照 PROP_ORDER[1:] 进行映射 (跳过 tag)
                for i, key in enumerate(self.PROP_ORDER):
                    if key == 'tag': continue
                    # 数据里的 index 0 对应 PROP_ORDER 的 index (如果是纯数字列表)
                    # PROP_ORDER: [tag, index, A, ...] -> raw_props: [index, A, ...]
                    # 所以 raw_props[0] -> index, raw_props[1] -> A
                    if i-1 < len(float_props):
                        properties[key].append(float_props[i-1])
                
                # 处理结构
                atoms = atoms_list[idx]
                # 伪造晶胞 (Fake Lattice) 适配 GNN
                atoms.set_cell([20.0, 20.0, 20.0])
                atoms.center()
                atoms.pbc = True 
                
                struct = AseAtomsAdaptor.get_structure(atoms)
                structures.append(struct)
                
                # 跳转
                current_line += (num_atoms + 2)
                idx += 1
                
            except ValueError:
                current_line += 1
                continue

        return structures, dict(properties)

    def _download_and_extract(self):
        if not osp.exists(self.root):
            os.makedirs(self.root)
        
        if not osp.exists(self.download_path):
            logger.info(f"Downloading {self.url}...")
            # 使用 paddle 或 urllib 下载
            import urllib.request
            urllib.request.urlretrieve(self.url, self.download_path)
            
        logger.info("Extracting data...")
        if self.download_path.endswith('tar.bz2') or self.download_path.endswith('tar.gz'):
            with tarfile.open(self.download_path, "r:*") as tar:
                tar.extractall(path=self.root)
        
        # 检查解压结果，确保 self.raw_xyz_path 存在
        # 如果解压出来是一堆小文件，可能需要这里做一个合并操作，
        # 但既然你给的是 paddle 的源，通常是处理好或者标准的合并版。
        if not osp.exists(self.raw_xyz_path):
             logger.warning(f"Expected {self.raw_xyz_path} not found. Please check extracted files.")

    def _load_data_references(self):
        """加载数据引用到内存"""
        prop_cache_dir = osp.join(self.cache_path, "properties")
        struct_cache_dir = osp.join(self.cache_path, "structures")
        graph_cache_dir = osp.join(self.cache_path, "graphs")
        
        # 加载所有需要的属性
        self.data_props = {}
        for pname in self.property_names:
            p_path = osp.join(prop_cache_dir, f"{pname}.pkl")
            if osp.exists(p_path):
                self.data_props[pname] = self._load_cache(p_path)
            else:
                raise FileNotFoundError(f"Property {pname} not found in cache.")
                
        # 加载结构列表 (只存路径，用到时再读，省内存)
        # 获取样本总数
        meta = self._load_cache(osp.join(self.cache_path, "meta.pkl"))
        self.num_samples = meta["num_samples"]
        
        self.structure_files = [osp.join(struct_cache_dir, f"{i:06d}.pkl") for i in range(self.num_samples)]
        
        # 加载图列表
        if self.build_graph_cfg is not None:
            self.graph_files = [osp.join(graph_cache_dir, f"{i:06d}.pkl") for i in range(self.num_samples)]
        else:
            self.graph_files = None

    def _filter_unvalid(self):
        """简单过滤无效数据 (NaN)"""
        valid_indices = []
        for i in range(self.num_samples):
            is_valid = True
            for pname in self.property_names:
                val = self.data_props[pname][i]
                if np.isnan(val) or np.isinf(val):
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(i)
        
        # 更新索引映射
        self.indices = valid_indices
        logger.info(f"Valid samples: {len(self.indices)} / {self.num_samples}")

    def _save_cache(self, path, data):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load_cache(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_structure_dict(self, structure: Structure) -> Dict[str, Any]:
        """将 Pymatgen Structure 转换为模型可用的 Numpy 字典"""
        atom_types = np.array([site.specie.Z for site in structure], dtype="int64")
        coords = structure.cart_coords.astype("float32")
        lattice = structure.lattice.matrix.astype("float32")
        
        return {
            "atom_types": atom_types,
            "coords": coords, # 用户指定用 coords 而不是 pos
            "lattice": lattice,
            "num_atoms": len(atom_types),
            # 虽然是分子，但为了兼容性设为 True 或根据模型需求设 False
            # 这里设为 True 配合 set_cell 的操作
            "pbc": np.array([True, True, True], dtype=bool) 
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 映射到真实索引
        real_idx = self.indices[idx]
        
        data = {}
        
        # 1. 获取图或结构
        if self.graph_files is not None:
            # 这是一个 Graph 对象 (通常是 paddle_geometric 的 Data 或类似)
            graph = self._load_cache(self.graph_files[real_idx])
            data["graph"] = graph
        else:
            struct = self._load_cache(self.structure_files[real_idx])
            # 将 Structure 对象转为 Tensor 字典
            struct_dict = self.get_structure_dict(struct)
            data.update(struct_dict)

        # 2. 获取属性
        for pname in self.property_names:
            val = self.data_props[pname][real_idx]
            # 转换为 (1,) 维度的 tensor 兼容格式
            data[pname] = np.array([val], dtype="float32")
            
        # 3. Apply Transforms
        if self.transforms is not None:
            data = self.transforms(data)
            
        return data