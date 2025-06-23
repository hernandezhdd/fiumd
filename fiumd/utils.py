import csv
import json
import os
import numpy as np
import time
import argparse
import yaml
import torch

def attrs_info(module):
    import inspect
    import torch.nn as nn

    print("=== Attributes ===\n")

    attrs = [
        attr for attr in dir(module)
        if not attr.startswith("_") and not callable(getattr(module, attr))
        # if not callable(getattr(module, attr))
    ]
    attrs.sort()
    max_attr_len = max((len(attr) for attr in attrs), default=0)

    for attr in attrs:
        value = getattr(module, attr)
        attr_fmt = attr.ljust(max_attr_len)

        if isinstance(value, torch.Tensor):
            print(f"{attr_fmt} : Tensor | shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
        elif isinstance(value, (list, tuple)):
            preview = f"{type(value).__name__} | length={len(value)}"
            if len(value) > 0:
                first = value[0]
                if isinstance(first, torch.Tensor):
                    preview += f", element[0]: shape={tuple(first.shape)}, dtype={first.dtype}"
                else:
                    preview += f", element[0]: type={type(first).__name__}"
            print(f"{attr_fmt} : {preview}")
        elif isinstance(value, dict):
            keys_preview = list(value.keys())[:5]
            suffix = "..." if len(value) > 5 else ""
            print(f"{attr_fmt} : Dict | keys={keys_preview}{suffix}")
        elif isinstance(value, nn.ModuleList):
            print(f"{attr_fmt} : ModuleList | length={len(value)}")
        else:
            print(f"{attr_fmt} : {type(value).__name__} | {value}")
            
                
def methods_info(module):
    import inspect
    import torch.nn as nn

    print("\n=== Public Methods ===")
    print("\n• Inherits all methods from nn.Module (e.g., `.to()`, `.eval()`, etc.)")

    cls = module.__class__
    base_methods = set(dir(nn.Module))
    own_methods = [
        (name, getattr(module, name))
        for name in dir(module)
        if not name.startswith("_")
        and callable(getattr(module, name))
        and name not in base_methods
    ]

    if not own_methods:
        print("  (none)")
    else:
        for name, method in sorted(own_methods):
            try:
                sig_obj = inspect.signature(method)
                params = str(sig_obj).split("->")[0].strip()
                return_ann = sig_obj.return_annotation
                return_str = ""
                if return_ann is not inspect.Signature.empty:
                    return_str = f"-> {return_ann}"
            except Exception:
                params = "(...)"
                return_str = ""
        
            doc = inspect.getdoc(method)
            doc_summary = doc.splitlines()[0] if doc else ""
            if len(doc_summary) > 80:
                doc_summary = doc_summary[:77] + "..."
        
            print(f"\n• {name}{params}")
            if return_str:
                print(f"    Returns{return_str}")
            if doc_summary:
                print(f"    {doc_summary}")

class LogWriter(object):
    # kind of inspired form openai.baselines.bench.monitor
    # We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, header="", name="monitor.csv"):
        self.keys = tuple(keys) + ("t",)
        assert path is not None

        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, name)
        if os.path.exists(filename):
            os.remove(filename)

        print("Writing logs to ", filename)

        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = "# {} \n".format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)
        self.logger.writeheader()
        self.f.flush()
        self.tstart = time.time()

    def write_row(self, epinfo):
        if self.logger:
            t = time.time() - self.tstart
            epinfo["t"] = t
            self.logger.writerow(epinfo)
            self.f.flush()


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
                return

        with values as f:
            input = f.read()
            input = input.rstrip()
            for lines in input.split("\n"):
                k, v = lines.split("=")
                typ = type(namespace.__dict__[k])
                v = typ(v) if typ is not None else v
                namespace.__dict__[k] = v


def save_argparse(args, filename, exclude=None):
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [
                exclude,
            ]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        with open(filename, "w") as fout:
            yaml.dump(args, fout)
    else:
        with open(filename, "w") as f:
            for k, v in args.__dict__.items():
                if k is exclude:
                    continue
                f.write(f"{k}={v}\n")


def xyz_writer(input_file, output_file, mol_elements):
    """Convert npy trajectory to xyz format
    Args:
    input_file (str): path to npy trajectory
    output_file (str): path to output xyz file
    mol_elements (list): list of elements in the molecule
    """
    npy_traj = np.load(input_file)
    Nats, _, Nsteps = npy_traj.shape
    with open(output_file, "a") as f:
        for i in range(Nsteps):
            f.write(f"{Nats}\n\n")
            for j in range(Nats):
                f.write(f"{mol_elements[j]} ")
                f.write(" ".join(map(str, npy_traj[j, :, i])))
                f.write("\n")
