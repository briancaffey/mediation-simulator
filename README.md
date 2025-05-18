# Mediation Simulator

Mediation Simulator helps develop your mediation skills, powered by NVIDIA Agent Intelligence Toolkit. This project is my submission for the NVIDIA Agent Intelligence Toolkit Hackathon.


## Case Generation

```
# add command
```


## Simulating Mediation for a Case

```
# add command
```

## Mediation Simulator Viewer

```
python -m http.server 8083
```

Visit [http://[::]:8083/](http://[::]:8083/)

## Milvus with docker compose

Milvus is used for indexing case documents and for parties to retrieve relevant documents during mediation.

```
docker compose -f docker-compose.milvus.yml up
```