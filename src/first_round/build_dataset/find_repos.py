import os
import argparse
from datetime import datetime
import pandas as pd
from github import Github
from tqdm import tqdm

ACCESS_TOKEN = os.getenv('GITHUB_ACCESS_TOKEN')

DATA_FIELDS = [
    "id", "clone_url", "created_at", "description", "full_name",
    "language", "name", "size", "stargazers_count", "updated_at", "forks_count"
]

github_client = Github(ACCESS_TOKEN)

def search_github(language, start_date, end_date):
    assert(language == "all" or language == "VHDL" or language == 'Verilog' or language == 'SystemVerilog')
    date_query = f"{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
    if language == "all":
        result = github_client.search_repositories(query=f"created:{date_query}")
    else:
        result = github_client.search_repositories(query=f"language:{language} created:{date_query}")
    return result

def add_repo_to_df(df, repo):
    if repo.stargazers_count >= 1:
        data = [getattr(repo, field) for field in DATA_FIELDS]
        df.loc[len(df)] = data
    return df

def process_repo_search_results(df, results):
    for i in range(results.totalCount//30+1):
        page = results.get_page(i)
        page_size = len(page)
        if page_size == 0:
            break
        print(f'Adding Page {i} to DataFrame...') 
        for j in tqdm(range(page_size)):
            df = add_repo_to_df(df, page[j])
    return df

def find_repos(df, language, start_date, end_date):
    repo_search_results = search_github(language, start_date, end_date)
    
    if repo_search_results.totalCount > 0:
        if repo_search_results.totalCount == 1000:
            delta = (end_date - start_date) / 2
            df = find_repos(df, language, start_date, end_date-delta)
            df = find_repos(df, language, end_date-delta, end_date)
        else:
            print(f"Finding {language} repo from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
            print(f'Adding {repo_search_results.totalCount} Repos to DataFrame ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})...') 
            df = process_repo_search_results(df, repo_search_results)
            print(f"Done: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, df length: {len(df)}")
    
    return df

def main(language):
    df = pd.DataFrame(columns=DATA_FIELDS)
    
    start_date = datetime(2010, 1, 1)
    end_date   = datetime(2023, 1, 1)

    df = find_repos(df, language, start_date, end_date)
    print(f"Done: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, df length: {len(df)}")
    
    df.to_csv(f"{language}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv")
    print('Write df to csv finish...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', help='download which language repo', default='Verilog')
    
    args = parser.parse_args()
    main(args.language)