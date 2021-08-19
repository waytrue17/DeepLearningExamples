import argparse
import os
import subprocess
import json

os.environ['NCCL_DEBUG'] = 'INFO'

def parse_args():
    parser = argparse.ArgumentParser(description='Get model info')
    parser.add_argument('--num_nodes', type=int, help='Number of nodes')
    # parser.add_argument('--data', type=str, help='Path to data')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--backbone', type=str, help='Backbone')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--warmup', type=int, help='Warmup')
    parser.add_argument('--bs', type=int, help='Batch size')
    parser.add_argument("--evaluation", type=int, help="Epochs at which to evaluate")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")

    args, _ = parser.parse_known_args()
    print (_)
    return args


def invoke_train(cmd):
    process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    args = parse_args()
    num_nodes = args.num_nodes
    # data = args.data
    epochs = args.epochs
    backbone = args.backbone
    learning_rate = args.learning_rate
    warmup = args.warmup
    bs = args.bs
    evaluation = args.evaluation
    seed = args.seed
    amp = '--amp' if args.amp else ''
    main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'main.py'))

    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    # work_dir = os.environ['SM_OUTPUT_DATA_DIR']
    work_dir = '/opt/ml/code'
    data_dir = os.environ["SM_CHANNEL_TRAIN"]

    cmd = f"python -m torch.distributed.launch --nnodes={num_nodes} --node_rank={rank} --nproc_per_node={num_gpus} \
        --master_addr={hosts[0]} --master_port='12345' \
    {main_path} --data {data_dir} --epochs {epochs} --backbone {backbone} --learning-rate {learning_rate} --warmup {warmup} --bs {bs} --evaluation {evaluation} --seed {seed} {amp}"

    print (cmd)
    invoke_train(cmd)
