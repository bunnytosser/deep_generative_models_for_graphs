import os
import pickle
import shutil
import torch
from dgl import model_zoo

from torch.utils.data import DataLoader
from dataset import DgmgDataset,DgmgPrinting
from utils import MoleculeDataset, set_random_seed, download_data,\
    mkdir_p, summarize_molecules, get_unique_smiles, get_novel_smiles
from rdkit import Chem
from featurizer import Featurizer
import pandas as pd
def generate_and_save(log_dir,scaffold, num_samples, max_num_steps, model,line_num,based_on):
    smiles=[]
    with open( 'results/generated_smiles%s.txt'%line_num, 'w') as f:
        f.write(based_on+"\n")
        for i in range(num_samples):
            for scf in scaffold:
                with torch.no_grad():
                    s = model(scaffold=scf,rdkit_mol=True, actions=None,max_num_steps=max_num_steps)
                smiles.append(s)
                f.write(s + '\n')
        generation_summary=summarize_molecules(smiles)
    print(generation_summary,type(generation_summary))
    gensum=pd.DataFrame(generation_summary)
    gensum.to_csv(os.path.join("results/summary_%s.csv"%line_num))
    with open("data/zinc.txt") as f:
        train_smiles = f.read().splitlines()
#    train_summary = summarize_molecules(train_smiles)
#    with open(os.path.join(args['log_dir'], 'train_summary.pickle'), 'wb') as f:
#    pickle.dump(train_summary, f)
    valid_generated_smiles = generation_summary['smile']
    unique_generated_smiles = get_unique_smiles(valid_generated_smiles)
    print("valid:",valid_generated_smiles)
    unique_train_smiles = get_unique_smiles(train_smiles)
    novel_generated_smiles = get_novel_smiles(unique_generated_smiles, unique_train_smiles)
    if len(valid_generated_smiles)!=0:
        with open('results/generation_stats_for_%s.txt'%line_num, 'w') as f:
            f.write('Total number of generated molecules: {:d}\n'.format(len(smiles)))
            f.write('Validity among all: {:.4f}\n'.format(
                len(valid_generated_smiles) / len(smiles)))
            f.write('Uniqueness among valid ones: {:.4f}\n'.format(
                len(unique_generated_smiles) / len(valid_generated_smiles)))
            f.write('Novelty among unique ones: {:.4f}\n'.format(
                len(novel_generated_smiles) / len(unique_generated_smiles)))
    else:
        print("no valid smiles generated")

def prepare_for_evaluation(rank, based_on,args):
    worker_seed = args['seed'] + rank * 10000
    set_random_seed(worker_seed)
    torch.set_num_threads(1)

    # Setup dataset and data loader
    scaffold = DgmgDataset("data/temp_scaffold.pkl")

    scaffold = DataLoader(scaffold, batch_size=args['batch_size'],shuffle=False, collate_fn=scaffold.collate_single)
    # Initialize model
    if not args['pretrained']:
        model = model_zoo.chem.DGMG(atom_types=["C","H","O","N","S","Cl","F","Br"],
                                    bond_types=[Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC],
                                    node_hidden_size=args['node_hidden_size'],
                                    num_prop_rounds=args['num_propagation_rounds'], dropout=args['dropout'])
        model.load_state_dict(torch.load(args['model_path'])['model_state_dict'])
    else:
        model = model_zoo.chem.load_pretrained('_'.join(['DGMG', args['dataset'], args['order']]), log=False)
    model.eval()
    print("finish evaluation")
    worker_num_samples = args['num_samples'] // args['num_processes']

    worker_log_dir = args['log_dir']
    print("log dir",worker_log_dir)
    
    generate_and_save(args['log_dir'],scaffold, worker_num_samples, args['max_num_steps'], model,args["line_num"],based_on)

def remove_worker_tmp_dir(args):
    for rank in range(args['num_processes']):
        worker_path = os.path.join(args['log_dir'], str(rank))
        try:
            shutil.rmtree(worker_path)
        except OSError:
            print('Directory {} does not exist!'.format(worker_path))

def aggregate_and_evaluate(args):
    print('Merging generated SMILES into a single file...')
    smiles = []
    for rank in range(args['num_processes']):
        with open(os.path.join(args['log_dir'], str(rank), 'generated_smiles.txt'), 'r') as f:
            rank_smiles = f.read().splitlines()
        smiles.extend(rank_smiles)

#    with open(os.path.join(args['log_dir'], 'generated_smiles.txt'), 'w') as f:
#        for s in smiles:
#            f.write(s + '\n')
#
    print('Removing temporary dirs...')
    remove_worker_tmp_dir(args)

    # Summarize training molecules
#    print('Summarizing training molecules...')
#    train_file = '_'.join([args['dataset'], 'DGMG_train.txt'])
#    if not os.path.exists(train_file):
#        download_data(args['dataset'], train_file)
#    with open(train_file, 'r') as f:
#        train_smiles = f.read().splitlines()
#    train_summary = summarize_molecules(train_smiles, args['num_processes'])
#    with open(os.path.join(args['log_dir'], 'train_summary.pickle'), 'wb') as f:
#        pickle.dump(train_summary, f)
#
    # Summarize generated molecules
#    print('Summarizing generated molecules...')
#    generation_summary = summarize_molecules(smiles, args['num_processes'])
#    with open(os.path.join(args['log_dir'], 'generation_summary.pickle'), 'wb') as f:
#        pickle.dump(generation_summary, f)
#
    # Stats computation
   # print('Preparing generation statistics...')
   # valid_generated_smiles = generation_summary['smile']
   # unique_generated_smiles = get_unique_smiles(valid_generated_smiles)
   # unique_train_smiles = get_unique_smiles(train_summary['smile'])
   # novel_generated_smiles = get_novel_smiles(unique_generated_smiles, unique_train_smiles)
   # with open(os.path.join(args['log_dir'], 'generation_stats.txt'), 'w') as f:
   #     f.write('Total number of generated molecules: {:d}\n'.format(len(smiles)))
   #     f.write('Validity among all: {:.4f}\n'.format(
   #         len(valid_generated_smiles) / len(smiles)))
   #     f.write('Uniqueness among valid ones: {:.4f}\n'.format(
   #         len(unique_generated_smiles) / len(valid_generated_smiles)))
   #     f.write('Novelty among unique ones: {:.4f}\n'.format(
   #         len(novel_generated_smiles) / len(unique_generated_smiles)))

if __name__ == '__main__':
    import argparse
    import datetime
    import time
    from rdkit import rdBase
    import tqdm
    from utils import setup

    parser = argparse.ArgumentParser(description='Evaluating DGMG for molecule generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--scaffold', type=str)
    # configure
    parser.add_argument('-s', '--seed', type=int, default=123, help='random seed')

    # dataset and setting
    parser.add_argument('-o', '--order', choices=['random', 'canonical'],
                        help='order to generate graphs, used for naming evaluation directory')

    # log
    parser.add_argument('-l', '--log_dir', default='./eval_results',
                        help='folder to save evaluation results')

    parser.add_argument('-p', '--model-path', type=str, default=None,
                        help='path to saved model')
    parser.add_argument('-pr', '--pretrained', action='store_true',
                        help='Whether to use a pre-trained model')
    parser.add_argument('-ns', '--num-samples', type=int, default=100000,
                        help='Number of molecules to generate')
    parser.add_argument('-mn', '--max-num-steps', type=int, default=400,
                        help='Max number of steps allowed in generated molecules to ensure termination')

    # multi-process
    parser.add_argument('-np', '--num-processes', type=int, default=32,
                        help='number of processes to use')
    parser.add_argument('-gt', '--generation-time', type=int, default=600,
                        help='max time (seconds) allowed for generation with multiprocess')

    parser.add_argument('-nl', '--line_num', type=int, default=1)
    args = parser.parse_args()
    args = setup(args, train=False)
    rdBase.DisableLog('rdApp.error')

    t1 = time.time()
    line_num=args["line_num"]
    inputfile=args["scaffold"]
    print(inputfile)
    pairs=[]
    based_on=""
    with open(inputfile) as f:
        for i,line in enumerate(f):
            if i==line_num:
                line=line.split("\t")[1].split(">>")
                pairs.append([line[0].strip(),line[1].strip()])
                based_on=line[1].strip()
    print(based_on)
    feat=Featurizer()
    feat(pairs)
    if args['num_processes'] == 1:
        prepare_for_evaluation(0, based_on,args)
    else:
        import multiprocessing as mp

        procs = []
        for rank in range(args['num_processes']):
            p = mp.Process(target=prepare_for_evaluation, args=(rank, args,))
            procs.append(p)
            p.start()

        while time.time() - t1 <= args['generation_time']:
            if any(p.is_alive() for p in procs):
                time.sleep(5)
            else:
                break
        else:
            print('Timeout, killing all processes.')
            for p in procs:
                p.terminate()
                p.join()

    t2 = time.time()
    print('It took {} for generation.'.format(
        datetime.timedelta(seconds=t2 - t1)))
   # aggregate_and_evaluate(args)
