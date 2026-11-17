# 实践二_朴素贝叶斯

本项目唯一数据集为 西瓜数据集3.0。
主模型仅使用密度和含糖率。离散拓展单独评估，不混入主模型。

## 运行
在本文件夹打开终端：
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
.\.venv\Scripts\python -m pytest -q
```
随机种子42。输出位于results，包含CSV、JSON、PNG图和实验报告.md。environment.json记录实际运行版本；精确复现参考requirements-lock.txt。

## 阅读顺序
先看results/实验报告.md，再看metrics.json及预测CSV，最后按图名查看结果图。
结果由程序真实运行生成，报告中的参数和指标也随运行更新。

## Git管理
项目提供.gitignore并建立本地Git历史。云端仓库地址尚未指定；自行创建仓库后执行：
```powershell
git remote add origin <你的仓库地址>
git push -u origin HEAD
```
涉及云端上传需使用自己的账户与目标仓库。
