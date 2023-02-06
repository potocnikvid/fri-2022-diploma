import requests
import tqdm
import os
from github import Github, GitRelease
from zipfile import ZipFile

def download_github_asset(asset_url, local_filename) -> str:
    github_token = os.environ.get("GITHUB_API_TOKEN")
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/octet-stream"
    }
    with requests.get(asset_url, stream=True, headers=headers) as r:
        r.raise_for_status()
        # content-length may be empty, default to 0
        file_size = int(r.headers.get('Content-Length', 0))
        bar_size = 1024
        # fetch 8 KB at a time
        chunk_size = 8192
        # how many bars are there in a chunk?
        chunk_bar_size = chunk_size / bar_size
        # bars are by KB
        num_bars = int(file_size / bar_size)

        pbar = tqdm.tqdm(
            disable=None,  # disable on non-TTY
            total=num_bars,
            unit='KB',
            desc='Downloading {}'.format(local_filename),
            leave=True  # progressbar stays
        )
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # we fetch 8 KB, so we update progress by +8x
                    pbar.update(chunk_bar_size)
        pbar.set_description('Downloaded {}'.format(local_filename))
        pbar.close()

        return local_filename

def get_github_asset_url() -> str:
    github_token = os.environ.get("GITHUB_API_TOKEN")
    g = Github(github_token)

    repo = g.get_repo("potocnikvid/nokdb")
    latest: GitRelease = repo.get_releases().get_page(0)[0]

    asset_url = latest.get_assets().get_page(0)[0].url
    return asset_url

def unzip(src, dest) -> None:
    with ZipFile(src, 'r') as zipObj:
        zipObj.extractall(dest)
