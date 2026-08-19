import sys
import pickle
from backend.rag.retrieval.retrieval import FastBM25
sys.modules['__main__'].FastBM25 = FastBM25
