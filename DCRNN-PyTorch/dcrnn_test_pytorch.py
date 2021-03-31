import argparse
import numpy as np
import os
import sys
import yaml

from libs.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_filename = supervisor_config['data'].get('graph_filename')
        adj_mx = load_graph_data(graph_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        acc, outputs = supervisor.evaluate_acc('test')
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/dcrnn_airspace_test.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
