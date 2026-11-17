"""输入检查、结果保存及图形工具。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RANDOM_STATE=42
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def validate_data(X,y):
    a=np.asarray(X); b=np.asarray(y)
    if a.ndim!=2 or len(a)==0 or a.shape[1]==0: raise ValueError('特征必须为非空二维数据')
    if b.ndim!=1 or len(a)!=len(b): raise ValueError('特征和标签的维度或样本数不一致')
    if not np.isin(b,[0,1]).all(): raise ValueError('标签必须为0或1')
    if len(np.unique(b))!=2: raise ValueError('训练数据必须包含两个类别')
    if not np.isfinite(a.astype(float)).all(): raise ValueError('特征包含缺失值或无穷值')

def evaluate(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),
        'f1_macro':float(f1_score(y,pred,average='macro',zero_division=0)),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'classification_report':classification_report(y,pred,labels=[0,1],output_dict=True,zero_division=0)}

def save_json(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')

def savefig(name):
    plt.tight_layout(); plt.savefig(OUT/name,dpi=160); plt.close()

def boundary(model,Xtrain,ytrain,Xtest,ytest,names,title,filename):
    allx=np.vstack([Xtrain,Xtest]); low=allx.min(axis=0); high=allx.max(axis=0)
    pad=np.maximum((high-low)*.12,.02)
    xx,yy=np.meshgrid(np.linspace(low[0]-pad[0],high[0]+pad[0],240),np.linspace(low[1]-pad[1],high[1]+pad[1],240))
    grid=pd.DataFrame(np.c_[xx.ravel(),yy.ravel()],columns=names)
    pp=model.predict_proba(grid)[:,list(model.classes_).index(1)].reshape(xx.shape)
    plt.figure(figsize=(8,6)); plt.contourf(xx,yy,pp,levels=np.linspace(0,1,11),cmap='RdBu',alpha=.35)
    if pp.min()<.5<pp.max(): plt.contour(xx,yy,pp,levels=[.5],colors='black')
    for c,color in [(0,'tab:red'),(1,'tab:blue')]:
        mask=np.asarray(ytrain)==c
        plt.scatter(np.asarray(Xtrain)[mask,0],np.asarray(Xtrain)[mask,1],c=color,s=25,label=f'训练 类别{c}')
    pred=model.predict(Xtest)
    plt.scatter(np.asarray(Xtest)[:,0],np.asarray(Xtest)[:,1],facecolors='none',edgecolors='black',s=70,label='测试样本')
    bad=pred!=np.asarray(ytest)
    plt.scatter(np.asarray(Xtest)[bad,0],np.asarray(Xtest)[bad,1],marker='x',c='black',s=80,label='误分类')
    if len(Xtest)<=10:
        for i,(xy,truth,pr) in enumerate(zip(np.asarray(Xtest),ytest,pred)):
            plt.annotate(f'{i+1}:真{truth}/预测{pr}',xy,fontsize=8,xytext=(3,5),textcoords='offset points')
    plt.xlabel(names[0]);plt.ylabel(names[1]);plt.title(title);plt.legend(fontsize=8)
    savefig(filename)
