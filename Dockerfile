FROM python:3
MAINTAINER Peter Georgeson "peter@supernifty.org"
RUN apt-get update -y
RUN pip install uv
COPY . /app
WORKDIR /app
RUN uv sync --frozen
ENTRYPOINT ["uv", "run"]
EXPOSE 5000
CMD ["main.py"]
