import ndjson
import collections
import plot
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import json
import common_comments2_functions
from sklearn.cluster import KMeans
import string
import pandas as pd
import praw
import json
import re
import time
import read_subreddit_data

import os
import requests
from praw.models import MoreComments
from prawcore import NotFound
from prawcore.exceptions import Forbidden

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


username = 'mylizard'


authorData = 'https://api.pushshift.io/reddit/search/comment/?'
authorData += 'author=' + username
r2 = requests.get(authorData)
while r2.status_code != 200:
    r2 = requests.get(authorData)
    time.sleep(0.5)  # trying to avoid rate limit
    print(r2.status_code)

totalSubreddits = []

authorComments = r2.json()
for sub in authorComments['data']:
    for i in range(0, sub['score']+1):
        totalSubreddits.append(sub['subreddit'])


user_dict = read_subreddit_data.get_user_dict(totalSubreddits)
        

print(user_dict)