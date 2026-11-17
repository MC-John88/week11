# 仅使用 load_breast_cancer 的逻辑回归实验。
import argparse, platform, sys, warnings
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from common import *

def build_model(C=1.0,penalty='l2',solver='liblinear'):
    if C<=0: raise ValueError('C必须大于0')
    return Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(C=C,penalty=penalty,solver=solver,max_iter=5000,random_state=42))])

def split_data(X,y):
    validate_data(X,y)
    return train_test_split(X,y,test_size=.2,stratify=y,random_state=42)

def predict_threshold(model,X,threshold=.5):
    if not 0<=threshold<=1: raise ValueError('阈值必须在0到1之间')
    prob=model.predict_proba(X)[:,list(model.classes_).index(1)]
    return (prob>=threshold).astype(int),prob

def main(threshold=None):
    ds=load_breast_cancer(as_frame=True);X=ds.data;y=ds.target
    summary={'dataset':'load_breast_cancer','shape':list(X.shape),'classes':{'0':'恶性','1':'良性'},'distribution':y.value_counts().sort_index().to_dict(),'missing':int(X.isna().sum().sum())}
    Xtr,Xte,ytr,yte=split_data(X,y)
    summary['train_size']=len(Xtr);summary['test_size']=len(Xte)
    pd.DataFrame({'sample_index':y.index,'split':np.where(y.index.isin(ytr.index),'train','test'),'label':y}).to_csv(OUT/'split.csv',index=False)
    y.value_counts().sort_index().plot.bar();plt.title('乳腺癌类别分布');plt.xlabel('0 恶性 / 1 良性');plt.ylabel('样本数');savefig('class_distribution.png')
    axes=X.iloc[:,:4].hist(figsize=(10,7),bins=25)
    for ax,name in zip(axes.ravel(),X.columns[:4]):
        ax.set_xlabel(name);ax.set_ylabel('样本数')
    plt.suptitle('乳腺癌特征直方图');savefig('feature_histograms.png')
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    base=build_model();base.fit(Xtr,ytr)
    baseline_pred,_=predict_threshold(base,Xte);summary['baseline']=evaluate(yte,baseline_pred)
    summary['baseline_cv_accuracy']=cross_val_score(build_model(),Xtr,ytr,cv=cv,scoring='accuracy').tolist()
    # 网格中的solver与penalty均为有效组合；整个Pipeline在每折内部拟合。
    grid=[{'model__C':[.001,.01,.1,1,10,100],'model__penalty':['l1','l2'],'model__solver':['liblinear']},
          {'model__C':[.001,.01,.1,1,10,100],'model__penalty':['l2'],'model__solver':['lbfgs']}]
    search=GridSearchCV(build_model(),grid,cv=cv,scoring='f1_macro',n_jobs=1,return_train_score=True)
    search.fit(Xtr,ytr);best=search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(OUT/'grid_search.csv',index=False)
    summary['best_params']=search.best_params_;summary['best_cv_f1_macro']=float(search.best_score_)
    rows=[]
    for C in [.001,.01,.1,1,10,100]:
        m=build_model(C=C); oof=cross_val_predict(m,Xtr,ytr,cv=cv,method='predict_proba')[:,1]
        train_model=build_model(C=C).fit(Xtr,ytr)
        trainacc=float((train_model.predict(Xtr)==ytr).mean())
        for t in [.3,.4,.5,.6,.7]:
            e=evaluate(ytr,(oof>=t).astype(int)); rows.append({'C':C,'threshold':t,'cv_accuracy':e['accuracy'],'cv_f1_macro':e['f1_macro'],'train_accuracy':trainacc})
    pd.DataFrame(rows).to_csv(OUT/'C_threshold_analysis.csv',index=False)
    oof=cross_val_predict(best,Xtr,ytr,cv=cv,method='predict_proba')[:,1]
    threshold_rows=[{'threshold':float(t),'cv_accuracy':float(((oof>=t)==ytr).mean()),'cv_f1_macro':float(f1_score(ytr,(oof>=t).astype(int),average='macro'))} for t in [.3,.4,.5,.6,.7]]
    selected=max(threshold_rows,key=lambda r:(r['cv_f1_macro'],-abs(r['threshold']-.5)))['threshold']
    used=selected if threshold is None else threshold
    summary['threshold_analysis']=threshold_rows;summary['cv_selected_threshold']=selected;summary['used_threshold']=used
    pred,prob=predict_threshold(best,Xte,used);summary['test']=evaluate(yte,pred)
    # AUC以恶性0为关注类别，使用恶性概率，避免把良性召回误读成癌症检出率。
    fpr,tpr,_=roc_curve((yte==0).astype(int),1-prob);summary['malignant_auc']=float(auc(fpr,tpr))
    plt.figure();plt.plot(fpr,tpr,label=f'恶性 AUC={summary["malignant_auc"]:.4f}');plt.plot([0,1],[0,1],'--');plt.xlabel('假阳性率');plt.ylabel('真阳性率');plt.title('测试集ROC 关注恶性');plt.legend();savefig('roc.png')
    pd.DataFrame({'sample_index':Xte.index,'true_label':yte,'prediction':pred,'p_malignant':1-prob,'p_benign':prob}).to_csv(OUT/'predictions.csv',index=False)
    scaler=best.named_steps['scaler'];coef=best.named_steps['model'].coef_[0]
    pd.DataFrame({'feature':X.columns,'standardized_coefficient':coef,'original_coefficient':coef/scaler.scale_}).to_csv(OUT/'coefficients.csv',index=False)
    summary['standardized_intercept']=float(best.named_steps['model'].intercept_[0]);summary['original_intercept']=float(best.named_steps['model'].intercept_[0]-np.sum(coef*scaler.mean_/scaler.scale_))
    names=['mean radius','mean texture'];two=build_model().fit(Xtr[names],ytr)
    boundary(two,Xtr[names],ytr,Xte[names],yte,names,'二维逻辑回归展示 不是30维主模型','decision_boundary.png')
    curve=pd.DataFrame(rows).query('threshold == 0.5')
    plt.figure();plt.semilogx(curve.C,curve.cv_f1_macro,'o-',label='训练集5折 F1 macro');plt.semilogx(curve.C,curve.train_accuracy,'s--',label='训练准确率');plt.xlabel('C 越小正则化越强');plt.ylabel('指标值');plt.title('C 与训练及验证表现');plt.legend();savefig('C_curve.png')
    save_json('metrics.json',summary)
    report=f"""# 实践一实验报告

唯一数据集：load_breast_cancer。569×30，0恶性212条，1良性357条，无缺失值。
训练455条，测试114条；分层划分，random_state=42。

## 原理与流程
逻辑回归将特征线性组合通过sigmoid转为类别概率。StandardScaler和模型放入Pipeline，每个交叉验证折只在自身训练部分学习标准化参数。
C是正则化强度的倒数，小C约束更强，可能欠拟合；大C约束较弱，需观察训练与验证差距，而不能单凭参数大小认定过拟合。
30个系数与截距决定良性类别的对数几率；标准化后系数每单位对应一个训练标准差。coefficients.csv同时给出原始单位系数。

## 参数选择和结果
默认模型测试准确率 {summary['baseline']['accuracy']:.4f}。
训练集5折网格搜索最佳参数 {search.best_params_}，最佳宏F1 {search.best_score_:.4f}。
训练集折外概率选择阈值 {selected}；本次使用 {used}。阈值针对良性概率，阈值升高会更容易判为恶性。
最终测试准确率 {summary['test']['accuracy']:.4f}，宏F1 {summary['test']['f1_macro']:.4f}，恶性ROC AUC {summary['malignant_auc']:.4f}。
混淆矩阵（行真实、列预测，顺序恶性0、良性1）：{summary['test']['confusion_matrix']}。
精确率是预测某类中正确的比例，召回率是真实某类被找出的比例，F1综合二者；AUC描述跨阈值的排序能力。

## 图表和限制
见class_distribution.png、feature_histograms.png、roc.png、C_curve.png及decision_boundary.png。
二维图仅使用mean radius、mean texture独立训练；主模型始终使用全部30个特征。
网格搜索分数经过参数选择可能偏乐观，阈值选择也使用训练数据；最终保留测试集作独立评价。本作业为教学实验，不用于临床诊断。
改进可使用嵌套交叉验证估计选择流程的泛化误差，并按误诊代价选择阈值。
"""
    (OUT/'实验报告.md').write_text(report,encoding='utf-8')
    save_json('environment.json',{'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()})
    print(f"逻辑回归完成 accuracy={summary['test']['accuracy']:.4f}, AUC={summary['malignant_auc']:.4f}")

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--threshold',type=float,default=None)
    try: main(parser.parse_args().threshold)
    except (ValueError,FileNotFoundError) as exc: raise SystemExit(f'输入错误：{exc}')
