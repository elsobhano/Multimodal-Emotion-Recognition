import pandas as pd
import os
import itertools
import re
import os
import contextlib
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional, Tuple, TypeVar
import requests
from torch.utils.model_zoo import tqdm
import zipfile
import warnings
warnings.filterwarnings('ignore')


def create_dataset_directory(root, dataset_name):
        
        filename_dict = {'train':'train.zip',
                         'dev':'dev.zip',
                         'test':'test.zip',
                         'MSCTD_data_train':'MSCTD_data_train.zip',
                         'MSCTD_data_dev':'MSCTD_data_dev.zip',
                         'MSCTD_data_test':'MSCTD_data_test.zip'}
        
        id_dict = {'train':'156yOz7M1sAfz4RK6OEQoPsaaPOMhrsT6',
                   'dev':'1URhTfBeUQiAmzb_2gxtn4MCXUxFywGnR',
                   'test':'1MbzM9Twe5KCWAKwZYvZO_OK-v8qRVSPF',
                   'MSCTD_data_train':'1AEwXhfMApCWzGyGr6KRzPtx7hBy3jZXt',
                   'MSCTD_data_dev': '1h3YnPZlIdSqPsggoms1zK2ZF5enSBcDH',
                   'MSCTD_data_test':'1pYWqfTd9rZuQuKgL0wv34Ybz5OINI6SV'}
        
        isExist = os.path.exists(os.path.join(root, dataset_name))
        
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(root, dataset_name))
            print("The " +dataset_name+ " directory is created!")
            print('Downloading ...')
            download_file_from_google_drive(id_dict['MSCTD_data_'+dataset_name],
                                            os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name]))
            download_file_from_google_drive(id_dict[dataset_name],
                                os.path.join(root, dataset_name, filename_dict[dataset_name]))
            
            _extract_zip(os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name]),os.path.join(root, dataset_name))
            _extract_zip(os.path.join(root, dataset_name, filename_dict[dataset_name]),os.path.join(root, dataset_name))
            
        if(dataset_name=='train'):
            list_path_out = [os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name].split('.')[0], 'english_'+dataset_name+'.txt'),
           os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name].split('.')[0], 'sentiment_'+dataset_name+'.txt'),
           os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name].split('.')[0], 'image_index_'+dataset_name+'.txt'),
           os.path.join(root, dataset_name, filename_dict[dataset_name].split('.')[0]+'_ende')]
        else:
            list_path_out = [os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name].split('.')[0], 'english_'+dataset_name+'.txt'),
           os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name].split('.')[0], 'sentiment_'+dataset_name+'.txt'),
           os.path.join(root, dataset_name, filename_dict['MSCTD_data_'+dataset_name].split('.')[0], 'image_index_'+dataset_name+'.txt'),
           os.path.join(root, dataset_name, filename_dict[dataset_name].split('.')[0])]
        
        return list_path_out


def download_file_from_google_drive(file_id: str, fpath: str):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
    """

    url = "https://drive.google.com/uc"
    params = dict(id=file_id, export="download")
    with requests.Session() as session:
        response = session.get(url, params=params, stream=True)

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        else:
            api_response, content = _extract_gdrive_api_response(response)
            token = "t" if api_response == "Virus scan warning" else None

        if token is not None:
            response = session.get(url, params=dict(params, confirm=token), stream=True)
            api_response, content = _extract_gdrive_api_response(response)

        if api_response == "Quota exceeded":
            raise RuntimeError(
                f"The daily quota of the file {fpath} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )

        _save_response_content(content, fpath)

    # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text
    if os.stat(fpath).st_size < 10 * 1024:
        with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:
            text = fh.read()
            # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
            if re.search(r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)", text):
                warnings.warn(
                    f"We detected some HTML elements in the downloaded file. "
                    f"This most likely means that the download triggered an unhandled API response by GDrive. "
                    f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "
                    f"the response:\n\n{text}"
                )

def _extract_gdrive_api_response(response, chunk_size: int = 32 * 1024) -> Tuple[bytes, Iterator[bytes]]:
    content = response.iter_content(chunk_size)
    first_chunk = None
    # filter out keep-alive new chunks
    while not first_chunk:
        first_chunk = next(content)
    content = itertools.chain([first_chunk], content)

    try:
        match = re.search("<title>Google Drive - (?P<api_response>.+?)</title>", first_chunk.decode())
        api_response = match["api_response"] if match is not None else None
    except UnicodeDecodeError:
        api_response = None
    return api_response, content


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def _extract_zip(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(
        from_path, "r", compression=zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)

def files_to_dataframe(text_add, sentiment_add, index_add):
    """
    This function convert all necessary .txt files to a single dataframe

    Args :
    text_add : .txt of chats.
    sentiment_add : .txt of emotions for each line.
    index_add : .txt of related images for each line.

    Output -> pd.dataframe

    """
    df_text = pd.read_csv(text_add, delimiter = "\r\t", header=None, engine='python')
    df_text.columns = ['text']

    df_sentiment = pd.read_csv(sentiment_add, delimiter = "\t", header=None)
    df_sentiment.columns = ['label']

    df_index = pd.read_csv(index_add, delimiter = "\t", header=None)
    df_index.columns = ['indexes']

    #concatenation
    lst = [df_text, df_sentiment, df_index]
    df_result = pd.concat(lst, axis =1)

    return df_result