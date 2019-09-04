
%% ���������ڻع�Ԥ��,��������120ά ���ά��3228
clear all
close all
clc
format compact
%% ��������
% c=xlsread('c.xls','A2:O121');
% g=xlsread('g.xls','A2:O3229');
% inpu=c;%����
% outpu =g;%���
% clear c g
% save data inpu outpu
load data
%% 
%% ��һ��
[input,inputns]=mapminmax(inpu,0,1);
[output,outputns]=mapminmax(outpu,0,1);%�����С��һ��
input=input';
output=output';%ת����DBN�ܹ�����ĸ�ʽ
%% �������ݼ� ����̫��
P=input;      %ѵ������
T=output;
P_test=input;%��������
T_test=output;

%% ѵ���������죬�ֿ飬����
numcases=15;%ÿ�����ݼ�����������
numdims=size(P,2);%���������Ĵ�С
numbatches=1;%��15������ �ֳ�1��  ÿ��15��
% ѵ������
for i=1:numbatches
    train=P((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train;
end%���ֺõ�10�����ݶ�����batchdata��

%% 2.ѵ��RBM
%% rbm����
maxepoch=20;%ѵ��rbm�Ĵ���
numhid=100; numpen=50; numpen2=300;%dbn������Ľڵ���
disp('����һ��3��������������DBN����������ȡ');
%% �޼ලԤѵ��
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d ',numdims,numhid);
restart=1;
rbm1;%ʹ��cd-kѵ��rbm��ע���rbm�Ŀ��Ӳ㲻�Ƕ�ֵ�ģ����������Ƕ�ֵ��
vishid1=vishid;hidrecbiases=hidbiases;


fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d ',numhid,numpen);
batchdata=batchposhidprobs;%����һ��RBM��������������Ϊ�ڶ���RBM ������
numhid=numpen;%��numpen��ֵ����numhid����Ϊ�ڶ���rbm������Ľڵ���
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d\n ',numpen,numpen2);%200-100
batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=numpen2;%������������Ľڵ���
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;

%%%% ��ѵ���õ�RBM��Ȩ�غ�ƫִ��ջ��DBN%%%%%%%%%
w1=[vishid1; hidrecbiases]; %12-100
w2=[hidpen; penrecbiases]; %100-50
w3=[hidpen2; penrecbiases2];%50-30

%% �мල�ع��ѵ��
%===========================ѵ������=====================================%
%==========DBN�޼ල������ȡ��������Ҫ�����мල�Ļع��==================%
%���ں���ƫִ������ʵ������Ӧ�ð���һ��ȫΪ1��������w0x0+w1x1+..+wnxn ����x0Ϊ1������ w0Ϊƫ��b
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
lamda=inf;%����ϵ��
OutputWeight=pinv(H'+1/lamda) *T';%��������ϵ��lamda��lamda=inf����û������
Y=(H' * OutputWeight)';   
%%%%%%%%%% ����ѵ��������Ҫ������ͼ����
% ����һ��
T=mapminmax('reverse',T,outputns);
Y=mapminmax('reverse',Y,outputns);


%===========================���Թ���=====================================%
%=======================================================================%
N2 = size(P_test,1);
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
%% ������
test = [P_test ones(N2,1)];
w1probs = 1./(1 + exp(-test*w1));
  w1probs = [w1probs  ones(N2,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); 
  w2probs = [w2probs ones(N2,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
H1 = w3probs';
TY=(H1' * OutputWeight)';                       %   TY: the actual output of the testing data
% ����һ��
T_test=mapminmax('reverse',T_test',outputns);
TY=mapminmax('reverse',TY,outputns);%�������
%%%%%%%%%% ������Խ��

error=TY-T_test;%�������
err=round(error);%�����ȡ��������
%mse(err) 
%% %%%%%%%%%%%�������%%%%%%%%%%%%%%%%%
fprintf('���Լ�����������\n');
fprintf('�������MSE\n');
MSE=mse(error)%������ԽС����Ԥ��Խ׼
%% �����µ����ݽ���Ԥ��  ����û���µ�����  ���Ծ��õ�15��
load data
input1=inpu(:,1);
%��һ��
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
% ����һ��

TY=mapminmax('reverse',TY,outputns);%�������

