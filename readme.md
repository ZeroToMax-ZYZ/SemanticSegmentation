
## 配置文件
默认配置文件：

```text
train_cfg/CamVid/cfg.yaml
```

可以通过 `--cfg` 指定其他 YAML/JSON 配置文件，也可以通过 `--opts` 临时覆盖参数。

## 全量训练

使用默认配置进行完整训练：

```bash
python train.py --cfg train_cfg/CamVid/cfg.yaml
```

训练产物会保存到：

```text
logs/logs_upload/<exp_name>/
logs/logs_weights/<exp_name>/
logs/logs_tensorboard/<exp_name>/
```

## 快速测试训练

推荐用 smoke test 确认数据、模型、loss、日志链路是否能跑通：

```bash
python train.py --cfg train_cfg/CamVid/cfg.yaml --smoke-test
```

也可以手动覆盖配置做一个小样本调试：

```bash
python train.py --cfg train_cfg/CamVid/cfg.yaml --opts epochs=1 batch_size=2 debug_mode=debug debug_max_train_samples=4 debug_max_val_samples=2
```

## TensorBoard

启动 TensorBoard：

```bash
tensorboard --logdir logs/logs_tensorboard
tensorboard --logdir logs/logs_tensorboard --port 6007
```

## 常用命令行覆盖示例

```bash
python train.py --cfg train_cfg/CamVid/cfg.yaml --opts optimizer.lr=0.001 epochs=20
```

```bash
python train.py --cfg train_cfg/CamVid/cfg.yaml --opts image_size=[360,480] batch_size=4
```
