import sys, os
sys.path.insert(0, '.')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from src.retrieval.embedder import search

print('--- UK Employment Rights ---')
r = search('unfair dismissal employee rights', country='UK', top_k=3)
for x in r:
    print(f"  {x['score']} | {x['source']}")

print('\n--- UK Housing Act 1996 ---')
r = search('landlord eviction notice period', country='UK', top_k=3)
for x in r:
    print(f"  {x['score']} | {x['source']}")

print('\n--- PK Labour Rights ---')
r = search('minimum wages worker rights Pakistan', country='PK', top_k=3)
for x in r:
    print(f"  {x['score']} | {x['source']}")