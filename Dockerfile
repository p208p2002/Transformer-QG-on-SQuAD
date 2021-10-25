# syntax=docker/dockerfile:1
FROM python:3.7.4
WORKDIR /app
COPY . .
RUN pip install torch==1.7.0
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y 
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python" ]
