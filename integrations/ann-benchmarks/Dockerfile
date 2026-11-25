FROM ann-benchmarks

RUN apt-get update && apt-get install -y g++ git
RUN pip3 install numpy numba faiss-cpu
RUN pip3 install git+https://github.com/Bryan-Disch/e8-rerank
RUN python3 -c 'import e8rerank, e8rerank.index as m; assert m._KERNEL is not None'
