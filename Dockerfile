FROM competition-hub-registry.cn-beijing.cr.aliyuncs.com/alimama-competition/bidding-results:base

# 设置工作目录为/root/biddingTrainEnv
WORKDIR /root/biddingTrainEnv

# 将当前目录内容复制到位于/root/biddingTrainEnv的容器中
COPY ./requirements.txt .

# 安装requirements.txt中指定的所有依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 将当前目录内容复制到位于/root/biddingTrainEnv的容器中
COPY . .

# 当容器启动时运行run_evaluate.py脚本
CMD ["python3", "./run/run_evaluate.py"]

ENV PYTHONPATH="/root/biddingTrainEnv:${PYTHONPATH}"
