import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_language_file(language):
    project_names = []

    file_name = f'{language}_project.txt'

    with open(file_name, 'r') as file:
        for line in file:
            project_names.append(line.strip())

    return project_names

def clone_repos(df, language):
    download_files = download_language_file(language)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, clone_url in tqdm(enumerate(df['clone_url'])):
            name = df['name'][i]
            if name not in download_files:
                executor.submit(os.system, (f'git clone {clone_url} ../../../data/first_round/origin_github_repos/{language}/{name}'))

def main(language):
    csv_file = f"{language}_2010-01-01_2023-01-01.csv"
    df = pd.read_csv(csv_file)
    clone_repos(df, language)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', help='download which language repo', default='Verilog')
    
    args = parser.parse_args()
    main(args.language)