#%%
import argparse

parser = argparse.ArgumentParser(description='Get hyperparameters')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--n_layers', type=int, default=1, help='number of classes')
parser.add_argument('--dim_ff', type=int, default=800, help='number of hidden units')
parser.add_argument('--n_heads', type=int, default=8, help='number of heads')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

args = parser.parse_args()
print(args.accumulate(args.integers))

# %%
