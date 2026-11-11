import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))


def main():
    parser = argparse.ArgumentParser(description='JEPA execution-world training and 200-task evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['verify-assets', 'prepare', 'train', 'evaluate', '_worker']:
        p = commands.add_parser(name)
        p.add_argument('--assets', type=Path, default=Path(__file__).parent / 'assets')
        if name != 'verify-assets':
            p.add_argument('--output', type=Path, required=True)
        if name == 'evaluate':
            p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
            p.add_argument('--checkpoint', type=Path)
            p.add_argument('--inputs', choices=['recorded', 'recorded-forced', 'fresh'], default='recorded')
            p.add_argument('--judge-workers', type=int, default=64)
            p.add_argument('--worker-retries', type=int, default=0,
                           help='requeue a failed shard (e.g. CUDA OOM from a shared GPU) up to N times')
        elif name == 'train':
            p.add_argument('--stage', choices=['initial', 'semantic', 'expanded', 'incumbent', 'all'], required=True)
            p.add_argument('--parent', type=Path)
            p.add_argument('--max-steps', type=int, default=0)
            p.add_argument('--no-development', action='store_true')
        elif name == '_worker':
            p.add_argument('--seed', type=int, required=True)
            p.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'verify-assets':
        from io_utils import verify_assets
        print(json.dumps(verify_assets(args.assets)))
    elif args.command == 'prepare':
        from preparation import prepare
        prepare(args)
    elif args.command == 'train':
        from training import run
        run(args)
    elif args.command == 'evaluate':
        from evaluation import run
        run(args)
    else:
        from evaluation import worker
        worker(args)


if __name__ == '__main__':
    main()
