import time
import ujson
import json
import orjson
import rapidjson

m = {
    "Title": "HOW TO USE THE BENCHMARK",
    "URL": "0ed1a1c3-050c-4fb9-9426-a7e72d0acfc7",
    "COSINE": "123",
    "PAGERANK": "123",
    "WIKI_DATA_ID": "started",
}

def benchmark(name, dumps):
    start = time.time()
    for i in range(1000000):
        dumps(m)
    print(name, time.time() - start)

benchmark("Python", json.dumps)
# orjson only outputs bytes, but often we need unicode:
benchmark("ujson", ujson.dumps)
benchmark("orjson", lambda s: str(orjson.dumps(s), "utf-8"))
benchmark("rapidjson", rapidjson.dumps)