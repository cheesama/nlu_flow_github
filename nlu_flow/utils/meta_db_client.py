## use https://strapi.io/ to manage meta DB or dataset
from tqdm import tqdm

import requests
import json
import ast
import os, sys

headers = {'accept': 'application/json', 'content-type': 'application/json'}

endpoint = os.getenv('META_ENDPOINT')

def request_to_server(
        method: str = "get", url: str = None, data: dict = None, headers: dict = None,
):
    assert method in ["get", "post", "put", "patch", "delete"]
    assert endpoint is not None

    data = json.loads(json.dumps(data))

    if method == "get":
        response = requests.get(endpoint + url, headers=headers)
    elif method == "put":
        response = requests.put(endpoint + url, headers=headers, json=data)
    elif method == "post":
        response = requests.post(endpoint + url, headers=headers, json=data)
    #
    # elif method == "patch":
    #     response = requests.patch(endpoint + url, headers=headers, params=params)
    #
    # elif method == "delete":
    #     response = requests.delete(endpoint + url, headers=headers)

    response.encoding = None
    try:
        return json.loads(response.text)
    except ValueError:
        return response.text


def validate_api_output(api_result):
    if api_result.__class__ == dict:
        statusCode = api_result.get('statusCode')
        if statusCode and statusCode != 200:
            return False
    return True

# get data from url(table)
def get(url: str):
    cnt = request_to_server('get', '{}/count'.format(url))
    if not validate_api_output(cnt):
        raise ConnectionError("Strapi api failed")
    if cnt == 'Not Found':
        raise ConnectionError("Strapi api failed")

    res = []
    start, iter_num = 0, 10
    len_ = 0
    while len_ < int(cnt):
        tmp_ids = request_to_server('get', url='{}?_start={}&_limit={}'.format(url, start, iter_num))
        res.extend(tmp_ids)
        start += iter_num
        len_ += len(tmp_ids)
    return res

def put(url, id, data):
    result = request_to_server('put', url=url + '/' + str(id), data=data)

def post(url, data):
    result = request_to_server('post', url=url, data=data)

#def get_regexs():
#    return get('regexes')

