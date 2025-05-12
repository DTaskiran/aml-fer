# Advanced Machine Learning FS25
## Facial Expression Recognition (FER)

## Setup Instructions

### 1. Install UV Package Manager

Choose one of the following methods:

```sh
# With curl 
curl -LsSf https://astral.sh/uv/install.sh | sh

# OR with brew
brew install uv
```

### 2. Configure UV autocomplete (for zsh) (Optional)
```sh
echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ~/.zshrc
```

### 3. Setup Project
```sh
git clone https://github.com/DTaskiran/aml-fer
uv sync
source .venv/bin/activate
```

### 4. Download Dataset
```sh
./download_data.sh
```

## Dataset
The project uses the FER-2013 facial expression dataset.  
Dataset Source: [Kaggle FER-2013](https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset)
