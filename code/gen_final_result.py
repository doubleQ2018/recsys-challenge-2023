import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--files", nargs="+", default=[])
args = parser.parse_args()

print('ensembling:', args.files)
result = pd.read_csv(args.files[0], sep='\t')
for f in args.files[1:]:
    df = pd.read_csv(f, sep='\t')
    result.is_installed += df.is_installed
result.is_installed = result.is_installed / len(args.files)

result.to_csv('result/submit.csv', index=False, sep='\t')
