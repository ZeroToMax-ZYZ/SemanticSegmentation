
## 配置文件

默认配置文件：

```text
train_cfg/CamVid/LeNet_cfg.yaml
```

可以通过 `--cfg` 指定其他 YAML 配置文件。如需快速修改参数，直接编辑 `train.py` 中的 `code_overrides` 字典即可。

## 训练

```bash
python train.py --cfg train_cfg/CamVid/LeNet_cfg.yaml
```

不传 `--cfg` 时默认使用 `train_cfg/CamVid/LeNet_cfg.yaml`。

训练产物保存到：

```text
logs/logs_upload/<exp_name>/       # CSV指标、曲线图、config.json
logs/logs_weights/<exp_name>/      # 模型权重（best/last/定期保存）
logs/logs_tensorboard/<exp_name>/  # TensorBoard日志
```

## TensorBoard

```bash
tensorboard --logdir logs/logs_tensorboard
```

## 数据增强可视化
```bash
python script/visualize_dataset.py --cfg train_cfg/CamVid/LeNet_cfg.yaml --num 6 --split val

```