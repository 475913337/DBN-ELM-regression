clear all
% %----------��ȡ����-----------------------
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
% %----------------���ݱ궨------------------------
% meas=(uel7b - uemp)./(ufu-uemp); %����ֵ��һ��
% 
% clear sensum;
% sensum_c1 = abs(sum(MSen));  % abs(sum()) or sum(abs())?
% M_SenC = zeros(n_meas, n_elem);%����һ��120*3228��ȫ����
% for i = 1:n_elem
%     M_SenC(:,i) = MSen(:,i)/sensum_c1(i);
% end
%  M_SenCt=M_SenC';
%  %-------------------Ƿ�������������------------------------
% %  x=M_SenCt*meas; %������ϵ����
% %_______________________Landweber������------------------
% % x=zeros(n_elem,1);%3228*1��ȫ����
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
%    %--------------�Ľ�������ֵ�ֽⷨ----------------------------------
% % x=zeros(n_elem,1);
% %[U,S,V]=svd(M_SenC);
% % s=S(1:120,1:120);
% % m=diag(s);
% % for i=1:120
% %    m(i)=m(i)/(m(i)*m(i)+0.0003);
% %end
% %m=diag(m);
% %n=zeros(3108,120);%3228��
% %c=[m;n];
% %x=V*c*U'*meas;
% %x=max(x,0);
%   %--------------Tikhonov����----------------------------------
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
%     %--------------ART�㷨----------------------------------
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
%     %--------------SIRT�㷨----------------------------------
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
%      %--------------ţ�ٵ���----------------------------------
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
%       %--------------CG�㷨----------------------------------
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


       %% ������ʼ��%
%����Ⱥ�㷨�е���������
% c1 = 2;
% c2 = 2;
% 
% M=100;   % ��������      
% N=40;   %��Ⱥ��ģ
% D=3228;
% Wmax=0.9;
% Wmin=0.6;
% popmax=255;
% popmin=0;
%  Vmax=0.65*popmax;
%   Vmin=0.65*popmin;
%% ������ʼ���Ӻ��ٶ�
% for i=1:N
%     %�������һ����Ⱥ
%     for j=1:D
%         x(i,j)=rand(1);
%         v(i,j)=rand(1);
%     end
%    %  y=x(i,:)*M_SenCt-meas';
%    % fun(i)=dot(y,y);%��Ӧ��
% end
%    pop(i,j)=0.15*rands(1,3228);    %��ʼ��Ⱥ
%    V(i,j)=0.2*rands(1,3228);        %��ʼ���ٶ�
    %������Ӧ��
   

%����õ�Ⱦɫ��
% [bestfun bestindex]=min(fun);
% pg=x(bestindex,:);   %ȫ�����
% p=x;                %�������
% fungbest=fun;     %���������Ӧ��ֵ
% funzbest=bestfun; %ȫ�������Ӧ��ֵ

%% ����Ѱ��
% for i=1:N
%     %��������
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
%         %�ٶȸ���
%        v(i,:) = v(i,:) + c1*rand*(y(i,:) - x(i,:)) + c2*rand*(pg - x(i,:));
%        v(i,find(v(i,:)>Vmax))=Vmax;
%        v(i,find(v(i,:)<Vmin))=Vmin;
%         
%         %��Ⱥ����
%        x(i,:)=x(i,:)+v(i,:);
%        x(i,find(x(i,:)>popmax))=popmax;
%        x(i,find(x(i,:)<popmin))=popmin;
%       
%       
%        % ��Ӧ��ֵ
% %        h=(x(i,:)*M_SenCt-meas');
% %        fun(i)=dot(h,h);
%        
%         
%         %�������Ÿ���
%         if p(i)<fun(pg,meas,M_SenCt)
%            p(i,:) = x(i,:);
%             fungbest(j) = fun(i);
%         end
%         
%         %Ⱥ�����Ÿ���
%         if fun(p(i),meas,M_SenCt) < fun(pg,meas,M_SenCt)
%            pg = x(i,:);
%            
%         end
%         
%     
%     yy(i)=pg;    
%         

% 
% % �������
% plot(yy,'Linewidth',2)
% title(['��Ӧ������  ' '��ֹ������' num2str(maxg)]);
% grid on
% xlabel('��������');ylabel('��Ӧ��');
% %������
% % ��Ѹ���ֵ
% X=zbest';
% x=max(X,0);
 %---------------ͼ����ʾ-------------------------------
 [XI,YI]=meshgrid(-46:0.1:46,-46:0.1:46);
 ZI=griddata(newnode(:,1),newnode(:,2),x,XI,YI,'cubic');%ɢ�ҵ��ֵ%���������ε����β岹��
 pcolor(XI,YI,ZI);
 shading interp;
 axis equal;
