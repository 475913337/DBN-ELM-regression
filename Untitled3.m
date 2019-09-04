clear all
% %----------读取数据-----------------------
% load cirs;%C125ufu
% MSen=cirs;
% load emp;%C125uemp
% uemp=meas;
% load meas1;%C125uel7b
% uel7b=meas;
% load ful;%C125MSen
% ufu=meas;
newnode=node_2D(46,64);
 load g;
x=sigmaxx;
% % load newnode125
% n_meas=120;
% n_elem=3228;
% %----------------数据标定------------------------
% meas=(uel7b - uemp)./(ufu-uemp); %电容值归一化
% 
% clear sensum;
% sensum_c1 = abs(sum(MSen));  % abs(sum()) or sum(abs())?
% M_SenC = zeros(n_meas, n_elem);%生成一个120*3228的全零阵
% for i = 1:n_elem
%     M_SenC(:,i) = MSen(:,i)/sensum_c1(i);
% end
%  M_SenCt=M_SenC';
%  %-------------------欠定性逆问题求解------------------------
% %  x=M_SenCt*meas; %灵敏度系数法
% %_______________________Landweber迭代法------------------
% % x=zeros(n_elem,1);%3228*1的全零阵
% % lan=0.008;
% % for k=1:100
% %     for j=1:n_meas
% %        e=M_SenC*x;   
% %        E(j)=meas(j)-e(j);
% %        ss=M_SenC(j,:)*M_SenCt(:,j);
% %        inc=M_SenCt(:,j)*E(j);
% %        x=x+lan*inc;
% %     end 
% % end   
% % x=max(x,0);
%    %----------------LBP--------------------------
%   X=zeros(n_elem,1);
%   X=M_SenC'*meas;
%    x=max(X,0);
%    %--------------改进后奇异值分解法----------------------------------
% % x=zeros(n_elem,1);
% %[U,S,V]=svd(M_SenC);
% % s=S(1:120,1:120);
% % m=diag(s);
% % for i=1:120
% %    m(i)=m(i)/(m(i)*m(i)+0.0003);
% %end
% %m=diag(m);
% %n=zeros(3108,120);%3228？
% %c=[m;n];
% %x=V*c*U'*meas;
% %x=max(x,0);
%   %--------------Tikhonov正则化----------------------------------
% %   x=zeros(n_elem,1);
% %   alpha=0.0003;
% %   I=eye(3228);
% %   C=M_SenC'*M_SenC+alpha*I;
% %   D=M_SenC'*meas;
% %   x=inv(C)*D;
% %   x=max(x,0);
%    %--------------GSVD-------------------
% % [X]=GSVD(M_SenC,meas,100)
% %  x=max(X,0);
%     %--------------ART算法----------------------------------
% %    x=zeros(n_elem,1);
% %     x=M_SenC'*meas;
% %     [m,n]=size(M_SenC);
% %    for k=1:2000
% %      for j=1:m
% %          a=M_SenC(j,:)*x-meas(j);
% %         b=M_SenC(j,:)*M_SenC(j,:)';
% %         x=x-a*M_SenC(j,:)'*inv(b);
% %       end
% %    end
% %     x=max(x,0);
%         
%     %--------------SIRT算法----------------------------------
% %      x=zeros(n_elem,1);
% %      x=M_SenC'*meas;
% %      diagS =diag(M_SenC*M_SenC');
% %      alpha = 0.0003;
% % %    for k=1:2000
% % %       for j=1:n_meas
% % %          e=M_SenC*x;   
% % %          E(j)=meas(j)-e(j);
% % %          ss=M_SenC(j,:)*M_SenCt(:,j);
% % %          inc=M_SenCt(:,j)*E(j)/diag(ss);
% % %          x=x+alpha*inc;
% % %       end 
% % %   end   
% % for i = 1 : 2000
% %    deltaC = meas - M_SenC* x;
% %    a = deltaC./diagS;
% %    x = x + alpha* M_SenC'*a ;
% % end
% % x=max(x,0);
%      %--------------牛顿迭代----------------------------------
% % x=zeros(n_elem,1);
% % x=M_SenC'*meas;
% % I=eye(3228);
% % alpha=0.0003;
% % C=M_SenC'*M_SenC+alpha*I;
% % for j=1:200
% %     deltaC=meas-M_SenC*x;
% %     x=x+inv(C)*M_SenC'*deltaC;
% % end
% % x=max(x,0);
%       %--------------CG算法----------------------------------
% A0=M_SenC'*M_SenC;
% b0=M_SenC'*meas;
% % [m,n] = size(A0);
% x=zeros(n_elem,1);
% x=M_SenC'*meas;
% r=b0-A0*x;
% p=r;
% for j=1:2000
%     z = A0*p;        
%     alpha = (r'*r)/(p'*z);   
%     x = x + alpha*p;    % x1 = x0 + alpha0*p0  
%     s = r'*r;           
%     r = r - alpha*A0*p;
%     beta = (r'*r)/s; 
%     p = r + beta*p;    % y is the search direction p
% end
% x=max(x,0);


       %% 参数初始化%
%粒子群算法中的两个参数
% c1 = 2;
% c2 = 2;
% 
% M=100;   % 进化次数      
% N=40;   %种群规模
% D=3228;
% Wmax=0.9;
% Wmin=0.6;
% popmax=255;
% popmin=0;
%  Vmax=0.65*popmax;
%   Vmin=0.65*popmin;
%% 产生初始粒子和速度
% for i=1:N
%     %随机产生一个种群
%     for j=1:D
%         x(i,j)=rand(1);
%         v(i,j)=rand(1);
%     end
%    %  y=x(i,:)*M_SenCt-meas';
%    % fun(i)=dot(y,y);%适应度
% end
%    pop(i,j)=0.15*rands(1,3228);    %初始种群
%    V(i,j)=0.2*rands(1,3228);        %初始化速度
    %计算适应度
   

%找最好的染色体
% [bestfun bestindex]=min(fun);
% pg=x(bestindex,:);   %全局最佳
% p=x;                %个体最佳
% fungbest=fun;     %个体最佳适应度值
% funzbest=bestfun; %全局最佳适应度值

%% 迭代寻优
% for i=1:N
%     %迭代次数
%     p(i)=fun(x(i,:),meas,M_SenCt);
%     y(i,:)=x(i,:);
% end
%     pg=x(N,:);
%     
%     for i=1:N-1
%         if fun(x(i,:),meas,M_SenCt)<fun(pg,meas,M_SenCt)
%             pg=x(i,:);
%         end
%     end
%     
%         
%         %速度更新
%        v(i,:) = v(i,:) + c1*rand*(y(i,:) - x(i,:)) + c2*rand*(pg - x(i,:));
%        v(i,find(v(i,:)>Vmax))=Vmax;
%        v(i,find(v(i,:)<Vmin))=Vmin;
%         
%         %种群更新
%        x(i,:)=x(i,:)+v(i,:);
%        x(i,find(x(i,:)>popmax))=popmax;
%        x(i,find(x(i,:)<popmin))=popmin;
%       
%       
%        % 适应度值
% %        h=(x(i,:)*M_SenCt-meas');
% %        fun(i)=dot(h,h);
%        
%         
%         %个体最优更新
%         if p(i)<fun(pg,meas,M_SenCt)
%            p(i,:) = x(i,:);
%             fungbest(j) = fun(i);
%         end
%         
%         %群体最优更新
%         if fun(p(i),meas,M_SenCt) < fun(pg,meas,M_SenCt)
%            pg = x(i,:);
%            
%         end
%         
%     
%     yy(i)=pg;    
%         

% 
% % 结果分析
% plot(yy,'Linewidth',2)
% title(['适应度曲线  ' '终止代数＝' num2str(maxg)]);
% grid on
% xlabel('进化代数');ylabel('适应度');
% %结果输出
% % 最佳个体值
% X=zbest';
% x=max(X,0);
 %---------------图像显示-------------------------------
 [XI,YI]=meshgrid(-46:0.1:46,-46:0.1:46);
 ZI=griddata(newnode(:,1),newnode(:,2),x,XI,YI,'cubic');%散乱点差值%基于三角形的三次插补法
 pcolor(XI,YI,ZI);
 shading interp;
 axis equal;
