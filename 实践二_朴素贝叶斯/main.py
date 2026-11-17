"""西瓜数据集3.0，仅密度与含糖率作为主模型特征。"""
import platform,sys
import sklearn
from sklearn.model_selection import train_test_split,LeaveOneOut,StratifiedKFold,cross_val_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from common import *
FEATURES=['密度','含糖率']

def build_model(priors=None,var_smoothing=1e-9):
    if var_smoothing<=0:raise ValueError('var_smoothing必须大于0')
    return Pipeline([('model',GaussianNB(priors=priors,var_smoothing=var_smoothing))])

def load_data():
    df=pd.read_csv(HERE/'watermelon3.0.csv')
    X=df[FEATURES];y=df['好瓜'].map({'否':0,'是':1})
    validate_data(X,y)
    return df,X,y

def split_data(X,y):
    validate_data(X,y)
    return train_test_split(X,y,test_size=.3,stratify=y,random_state=42)

def main():
    df,X,y=load_data();Xtr,Xte,ytr,yte=split_data(X,y)
    summary={'dataset':'西瓜数据集3.0','shape':[len(df),len(df.columns)-1],'main_features':FEATURES,'distribution':y.value_counts().sort_index().to_dict(),'missing':int(df.isna().sum().sum()),'label_mapping':{'0':'坏瓜','1':'好瓜'},'train_size':len(Xtr),'test_size':len(Xte)}
    pd.DataFrame({'row_number':y.index+1,'split':np.where(y.index.isin(ytr.index),'train','test'),'label':y}).to_csv(OUT/'split.csv',index=False)
    y.value_counts().sort_index().plot.bar();plt.xlabel('0 坏瓜 / 1 好瓜');plt.ylabel('样本数');plt.title('西瓜类别分布');savefig('class_distribution.png')
    plt.figure()
    for c in [0,1]:plt.scatter(X.loc[y==c,'密度'],X.loc[y==c,'含糖率'],label='好瓜' if c else '坏瓜')
    plt.xlabel('密度');plt.ylabel('含糖率');plt.title('西瓜密度与含糖率');plt.legend();savefig('feature_scatter.png')
    model=build_model().fit(Xtr,ytr);gnb=model.named_steps['model']
    pred=model.predict(Xte);prob=model.predict_proba(Xte);summary['test']=evaluate(yte,pred)
    summary['parameters']={'classes':gnb.classes_.tolist(),'theta_':gnb.theta_.tolist(),'var_':gnb.var_.tolist(),'class_prior_':gnb.class_prior_.tolist(),'epsilon_':float(gnb.epsilon_)}
    pd.DataFrame({'row_number':Xte.index+1,'true_label':yte,'prediction':pred,'p_bad':prob[:,0],'p_good':prob[:,1]}).to_csv(OUT/'predictions.csv',index=False)
    sample=pd.DataFrame([[.697,.460]],columns=FEATURES)
    summary['测1']={'features':[.697,.460],'prediction':int(model.predict(sample)[0]),'probabilities':model.predict_proba(sample)[0].tolist(),'in_training':0 in Xtr.index}
    # 独立手算校验：使用训练集MLE均值方差和与GaussianNB相同的方差平滑。
    scores=[]
    for i,c in enumerate(gnb.classes_):
        subset=Xtr.loc[ytr==c].to_numpy();mu=subset.mean(axis=0);var=subset.var(axis=0)+gnb.epsilon_
        scores.append(np.log(gnb.class_prior_[i])+np.sum(-.5*np.log(2*np.pi*var)-(sample.to_numpy()[0]-mu)**2/(2*var)))
    weights=np.exp(np.array(scores)-max(scores)); manual=weights/weights.sum()
    summary['测1']['manual_posterior']=manual.tolist()
    assert np.allclose(manual,model.predict_proba(sample)[0])
    boundary(model,Xtr,ytr,Xte,yte,FEATURES,'GaussianNB决策边界 0坏瓜 1好瓜','decision_boundary.png')
    loo=LeaveOneOut();five=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cvout={}
    for name,cv in [('LeaveOneOut',loo),('StratifiedKFold5',five)]:
        scores=cross_val_score(build_model(),X,y,cv=cv,scoring='accuracy')
        cvout[name]={'scores':scores.tolist(),'mean':float(scores.mean()),'fold_variance':float(scores.var()),'fold_std':float(scores.std())}
    summary['full_dataset_cv']=cvout
    trials=[]
    for priors in [None,[.5,.5]]:
        for smoothing in [1e-12,1e-9,1e-6,.001,.01,.1,1.0]:
            m=build_model(priors,smoothing).fit(Xtr,ytr);pp=m.predict_proba(Xte)
            scores=cross_val_score(build_model(priors,smoothing),Xtr,ytr,cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42),scoring='accuracy')
            trials.append({'priors':'empirical' if priors is None else 'equal','var_smoothing':smoothing,'train_cv_accuracy':scores.mean(),'test_accuracy':accuracy_score(yte,m.predict(Xte)),'test_p_good':pp[:,1].tolist()})
    pd.DataFrame(trials).to_csv(OUT/'prior_smoothing_analysis.csv',index=False)
    summary['train_cv_preferred_setting']=max(trials,key=lambda t:t['train_cv_accuracy'])
    # 全数据CV曲线是补充评估，不用于宣称独立测试调优效果。
    curve=[]
    for v in np.logspace(-12,0,13):
        a=cross_val_score(build_model(var_smoothing=v),X,y,cv=loo,scoring='accuracy');b=cross_val_score(build_model(var_smoothing=v),X,y,cv=five,scoring='accuracy')
        curve.append({'var_smoothing':v,'loo_accuracy':a.mean(),'fivefold_accuracy':b.mean()})
    pd.DataFrame(curve).to_csv(OUT/'smoothing_cv.csv',index=False)
    plt.figure();plt.semilogx([r['var_smoothing'] for r in curve],[r['loo_accuracy'] for r in curve],'o-',label='留一法');plt.semilogx([r['var_smoothing'] for r in curve],[r['fivefold_accuracy'] for r in curve],'s-',label='5折均值');plt.xlabel('var_smoothing');plt.ylabel('交叉验证准确率');plt.title('方差平滑与交叉验证表现');plt.legend();savefig('smoothing_cv.png')
    for priors,v,name in [(None,.1,'smoothing_boundary.png'),([.5,.5],1e-9,'equal_prior_boundary.png')]:
        boundary(build_model(priors,v).fit(Xtr,ytr),Xtr,ytr,Xte,yte,FEATURES,f'先验={priors} 平滑={v}',name)
    # 可选拓展①：取值集合来自预先定义的领域知识；编码器只拟合训练集。
    cols=['色泽','根蒂','敲声','纹理','脐部','触感']
    categories=[['青绿','乌黑','浅白'],['蜷缩','稍蜷','硬挺'],['浊响','沉闷','清脆'],['清晰','稍糊','模糊'],['凹陷','稍凹','平坦'],['硬滑','软粘']]
    cat=Pipeline([('encode',OrdinalEncoder(categories=categories)),('model',CategoricalNB(alpha=1,min_categories=[3,3,3,3,3,2]))])
    cat.fit(df.loc[Xtr.index,cols],ytr);summary['categorical_extension']=evaluate(yte,cat.predict(df.loc[Xte.index,cols]))
    save_json('metrics.json',summary)
    report=f"""# 实践二实验报告

唯一数据集：西瓜数据集3.0，17×8，8好瓜、9坏瓜，无缺失值。主模型仅用密度与含糖率。
好瓜1、坏瓜0；分层70/30，训练11条、测试6条，random_state=42。

## 模型原理及参数
GaussianNB假设给定类别后，两个特征相互独立，且每个类别下各特征服从高斯分布。
先验乘以两个高斯密度得到分数，再归一化成后验概率。theta_是类别均值，现代sklearn的var_是方差（教材旧称sigma_），class_prior_是先验；详细数值见metrics.json。
var_smoothing把训练数据最大特征方差的一定比例加到每个类别方差上，避免方差为零。它不是离散计数的拉普拉斯加一修正；高斯密度在数值计算中仍可能下溢，因此通常使用对数计算。

## 本次结果
测试准确率 {summary['test']['accuracy']:.4f}，宏F1 {summary['test']['f1_macro']:.4f}。
混淆矩阵行真实、列预测，顺序坏瓜0、好瓜1：{summary['test']['confusion_matrix']}。
测1预测 {summary['测1']['prediction']}，坏瓜/好瓜概率 {summary['测1']['probabilities']}；是否参与训练：{summary['测1']['in_training']}。
手工公式与模型概率已逐项核对。教材若使用全部17条及不同方差约定，手算数值不会与本次训练子集完全相同；测1若在训练集中，其预测不能算独立测试成绩。
全数据留一法均值 {cvout['LeaveOneOut']['mean']:.4f}，折分数方差 {cvout['LeaveOneOut']['fold_variance']:.4f}。
全数据5折均值 {cvout['StratifiedKFold5']['mean']:.4f}，折分数方差 {cvout['StratifiedKFold5']['fold_variance']:.4f}。
留一法每折只测1条，得分只能0或1；5折每折测3或4条。两种折分数方差不可直接解释成估计置信度。

## 参数影响与局限
prior_smoothing_analysis.csv比较经验/等概率先验和多种平滑参数，保留每条测试样本的好瓜概率。
候选参数优先按训练集3折均值排序：{summary['train_cv_preferred_setting']}。小数据下并列最优和划分波动很常见。
测试集参数扫描仅作敏感性展示，没有用测试准确率挑选并重报主模型；主模型保持默认GaussianNB。
先验改变后验偏好；方差平滑改变概率及边界。GaussianNB两类方差不同时，二维边界通常是二次曲线，不必是直线。
可选离散拓展使用OrdinalEncoder与CategoricalNB，测试准确率 {summary['categorical_extension']['accuracy']:.4f}。离散整数是类别编号，不代表大小，适合CategoricalNB；将其直接交给GaussianNB会引入不合适的高斯数值假设。
样本太少，应增加数据并通过多次分层评估检查稳定性；交叉验证也不能弥补代表性不足。
"""
    (OUT/'实验报告.md').write_text(report,encoding='utf-8')
    save_json('environment.json',{'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()})
    print(f"朴素贝叶斯完成 accuracy={summary['test']['accuracy']:.4f}, LOO={cvout['LeaveOneOut']['mean']:.4f}")

if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as exc:raise SystemExit(f'输入错误：{exc}')
