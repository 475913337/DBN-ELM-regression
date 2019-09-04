
%% 本程序用于回归预测,输入特征120维 输出维度3228
clear all
close all
clc
format compact
%% 加载数据
% c=xlsread('c.xls','A2:O121');
% g=xlsread('g.xls','A2:O3229');
% inpu=c;%输入
% outpu =g;%输出
% clear c g
% save data inpu outpu
load data
%% 
%% 归一化
[input,inputns]=mapminmax(inpu,0,1);
[output,outputns]=mapminmax(outpu,0,1);%最大最小归一化
input=input';
output=output';%转换成DBN能够处理的格式
%% 划分数据集 数据太少
P=input;      %训练输入
T=output;
P_test=input;%测试输入
T_test=output;

%% 训练样本构造，分块，批量
numcases=15;%每块数据集的样本个数
numdims=size(P,2);%单个样本的大小
numbatches=1;%就15个样本 分成1批  每批15个
% 训练数据
for i=1:numbatches
    train=P((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train;
end%将分好的10组数据都放在batchdata中

%% 2.训练RBM
%% rbm参数
maxepoch=20;%训练rbm的次数
numhid=100; numpen=50; numpen2=300;%dbn隐含层的节点数
disp('构建一个3层的深度置信网络DBN用于特征提取');
%% 无监督预训练
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d ',numdims,numhid);
restart=1;
rbm1;%使用cd-k训练rbm，注意此rbm的可视层不是二值的，而隐含层是二值的
vishid1=vishid;hidrecbiases=hidbiases;


fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d ',numhid,numpen);
batchdata=batchposhidprobs;%将第一个RBM的隐含层的输出作为第二个RBM 的输入
numhid=numpen;%将numpen的值赋给numhid，作为第二个rbm隐含层的节点数
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d\n ',numpen,numpen2);%200-100
batchdata=batchposhidprobs;%显然，将第二个RBM的输出作为第三个RBM的输入
numhid=numpen2;%第三个隐含层的节点数
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;

%%%% 将训练好的RBM的权重和偏执堆栈成DBN%%%%%%%%%
w1=[vishid1; hidrecbiases]; %12-100
w2=[hidpen; penrecbiases]; %100-50
w3=[hidpen2; penrecbiases2];%50-30

%% 有监督回归层训练
%===========================训练过程=====================================%
%==========DBN无监督用于提取特征，需要加上有监督的回归层==================%
%由于含有偏执，所以实际数据应该包含一列全为1的数，即w0x0+w1x1+..+wnxn 其中x0为1的向量 w0为偏置b
N1 = size(P,1);
digitdata = [P ones(N1,1)];
w1probs = 1./(1 + exp(-digitdata*w1));
w1probs = [w1probs  ones(N1,1)];
w2probs = 1./(1 + exp(-w1probs*w2));
w2probs = [w2probs ones(N1,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
H = w3probs'; 
nn=size(T,2);
T=T';
lamda=inf;%正则化系数
OutputWeight=pinv(H'+1/lamda) *T';%加入正则化系数lamda，lamda=inf就是没有正则化
Y=(H' * OutputWeight)';   
%%%%%%%%%% 计算训练误差，不重要，看看图就行
% 反归一化
T=mapminmax('reverse',T,outputns);
Y=mapminmax('reverse',Y,outputns);


%===========================测试过程=====================================%
%=======================================================================%
N2 = size(P_test,1);
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
%% 计算结果
test = [P_test ones(N2,1)];
w1probs = 1./(1 + exp(-test*w1));
  w1probs = [w1probs  ones(N2,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); 
  w2probs = [w2probs ones(N2,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
H1 = w3probs';
TY=(H1' * OutputWeight)';                       %   TY: the actual output of the testing data
% 反归一化
T_test=mapminmax('reverse',T_test',outputns);
TY=mapminmax('reverse',TY,outputns);%网络输出
%%%%%%%%%% 计算测试结果

error=TY-T_test;%计算误差
err=round(error);%对误差取四舍五入
%mse(err) 
%% %%%%%%%%%%%结果分析%%%%%%%%%%%%%%%%%
fprintf('测试集输出结果分析\n');
fprintf('均方误差MSE\n');
MSE=mse(error)%均方差越小，则预测越准
%% 输入新的数据进行预测  由于没有新的数据  所以就用第15组
load data
input1=inpu(:,1);
%归一化
input1n=mapminmax('apply',input1,inputns);
input1n=input1n';
N2 = size(input1n,1);
test = [input1n ones(N2,1)];
w1probs = 1./(1 + exp(-test*w1));
  w1probs = [w1probs  ones(N2,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); 
  w2probs = [w2probs ones(N2,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
H1 = w3probs';
TY=(H1' * OutputWeight)';                       %   TY: the actual output of the testing data
% 反归一化

TY=mapminmax('reverse',TY,outputns);%网络输出

