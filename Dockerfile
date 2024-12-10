FROM competition-hub-registry.cn-beijing.cr.aliyuncs.com/alimama-competition/bidding-results:base

WORKDIR /root/biddingTrainEnv

COPY ./requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "./run/run_evaluate.py"]

ENV PYTHONPATH="/root/biddingTrainEnv:${PYTHONPATH}"
