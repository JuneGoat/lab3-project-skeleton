# Lab - Setup a project from scratch

## Quickstart

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

默认会从 `./data/tiny-imagenet-200` 读取数据（需要包含 `train/` 子目录）。

也可以通过环境变量指定数据路径：

```bash
export DATA_ROOT=/path/to/tiny-imagenet-200
```

如果暂时没有数据集，但想先跑通训练流程，可以使用 FakeData：

```bash
export USE_FAKE_DATA=1
```
