# Mediation Simulator

Mediation Simulator helps develop your mediation skills, powered by NVIDIA Agent Intelligence Toolkit. This project is my submission for the NVIDIA Agent Intelligence Toolkit Hackathon.

![p](/flux/data/1747700517310.png)

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

## Image Generation with Flux

```
python flux/main.py
```


## Generate all_cases.yml for website

```
python aiq/generate_case_list_data.py
```