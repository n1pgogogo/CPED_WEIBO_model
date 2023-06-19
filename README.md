# 这是?

这里是模型，其中包含了测试部分和训练部分，其中data里只有200个用户。

只不过你可以使用`public_user.csv`使用全部的数据

# 如何使用

首先，你需要先安装环境
```shell
pip install -r requirements.txt
```

然后，运行`python pre.py`，对于数据进行预处理。

预处理完毕，你能够在data文件夹下面得到一个`pth`的文件。在训练和测试的时候都会用到。

## 训练
```
python train2.py
```
## 测试
```
python test.py
```