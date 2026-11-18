"""Plan original workflows, or run them in a separate working copy."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from leafshift_cp.workflows import WORKFLOWS, plan, run

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['list','plan','run'])
    parser.add_argument('workflow',choices=sorted(WORKFLOWS),nargs='?')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-expensive',action='store_true')
    parser.add_argument('--inputs-from',type=Path)
    parser.add_argument('--dataset',type=Path)
    args=parser.parse_args()
    if args.action=='list':
        print('\n'.join(sorted(WORKFLOWS)))
    elif not args.workflow:
        parser.error('A workflow name is required')
    elif args.action=='plan':
        print(json.dumps(plan(args.workflow),indent=2))
    else:
        if not args.output:parser.error('--output is required for isolated execution')
        print(json.dumps(run(args.workflow,args.output,allow_expensive=args.allow_expensive,inputs_from=args.inputs_from,dataset=args.dataset),indent=2))
