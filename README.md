# week11

第11周机器学习实践，包含两项相互独立、已运行的实验。

| 项目 | 唯一数据集及主模型 | 本次结果 | 自动化测试 |
| --- | --- | --- | --- |
| [实践一 逻辑回归](实践一_逻辑回归/README.md) | load_breast_cancer，30特征，StandardScaler + LogisticRegression | 测试准确率98.25%，恶性AUC 0.9960 | 10项通过 |
| [实践二 朴素贝叶斯](实践二_朴素贝叶斯/README.md) | 西瓜数据集3.0，仅密度、含糖率，GaussianNB | 测试准确率50%，留一法准确率64.71% | 9项通过 |

随机种子统一为42。以上是本次固定划分的结果，西瓜样本只有17条，不宜将单次测试准确率作为稳定的泛化性能估计。

## 查看报告

- [实践一实验报告](实践一_逻辑回归/results/实验报告.md)
- [实践二实验报告](实践二_朴素贝叶斯/results/实验报告.md)

每项的 results/实验报告.html 包含完整图表，下载后用浏览器打开。GitHub可直接查看PNG、Markdown、CSV及JSON结果。

## 运行

进入其中一个项目目录，在终端执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python main.py
.\.venv\Scripts\python -m pytest -q
```

锁定依赖记录本次运行版本，使用Python 3.12。通用安装约束见requirements.txt。各项目README提供更多说明。

## 输出与方法

每项均提供源码、依赖清单、数据探索、数据划分记录、预测表、评价指标、参数分析、结果图和测试记录。
逻辑回归只用训练数据选择参数和阈值，保留测试集作最终评估；二维模型单独训练，仅供展示。
西瓜主流程使用70/30分层划分，另提供留一法、5折评估、先验和平滑分析，以及CategoricalNB离散特征拓展。
乳腺癌实验仅用于教学，不能替代临床诊断。
