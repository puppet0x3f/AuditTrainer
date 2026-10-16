"""Fresh internal A/S values under the validated compute SFT executor."""
import argparse
from pathlib import Path
import sys
ROOT=Path('runs/refinement/compute_refresh')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from . import refinement_compute_values as probe
from .refinement_compute_sft import context


def prepare():
    probe.prepare(root=ROOT, prior_path='runs/refinement/compute_sft/request.json',
                  executor_adapter='runs/refinement/compute_sft/training/final_adapter',
                  plan_seed=461000713, execution_seed=462000713)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':probe.run(a.index,root=ROOT,policy_context=context)
    else:probe.report(root=ROOT)
