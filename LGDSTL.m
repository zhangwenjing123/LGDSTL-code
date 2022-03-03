function [W] = LGDSTL(Xs,Xt,Ys,Yt, param, opts)
alpha = param.alpha;
beta = param.beta; 
gamma = param.gamma;
lambda =param.lambda;
lambda1 =param.lambda1;
X = [Xs,Xt];
[m,n1] = size(Xs); 
[m,n2] = size(Xt);
[m,nn] = size(X);
dim = param.dim;
n = length(Ys);
mu = opts.mu;
rho = opts.rho;
mu_max = 1e8;
epsilon = 1e-7;
Max_Iter = opts.Max_Iter;
I=eye(max(Ys));
Y = I(:,Ys);
options.ReducedDim = dim;
[P1,~] = PCA1(Xs', options);
W = ones(size(Xs,1),dim);
v=sqrt(sum(W.*W,2)+eps);
D=diag(1./(v));
V = Y;
U = V;
Z = zeros(size(V));
F = zeros(size(X));
YY = Y'*Y;
XX = Xs*Xs';
%Graph
% Construct local_graph
options1.ReduceDim =100;
[P2,~] = PCA1(Xs', options1);
[Yt0,PCAAcc] = myClassifier(Xs,Xt,Ys,Yt,P2);
%fprintf('PCA accuracy : %f\n',PCAAcc);
accVec = zeros(Max_Iter,1);
[Lws,Lbs] = myConGraph2(Ys,options,Xs');
Sbs = Xs*Lbs*Xs';
Sws = Xs*Lws*Xs';
Ls = Lws - lambda*Lbs;
% Constuct global_graph
   options2 = [];
   options2.NeighborMode = 'KNN';
   options2.k =1;
   options2.WeightMode = 'Binary';
   W3 = constructW(X',options2);
   W3=full(W3);
   DD3 = diag(sum(W3));
   Lw2 = DD3 - W3;
   lg = X*Lw2*X';
%fprintf('PRDR:  alpha=%f  beta=%f  gamma=%f,lambda=%f,lambda1=%f\n',alpha,beta,gamma,lambda,lambda1);
for iter=1:Max_Iter
     [Lwt,Lbt] = myConGraph2(Yt0,options,Xt');
     Sbt = Xt*Lbt*Xt';
     Swt = Xt*Lwt*Xt'; 
     Lt = Lwt -lambda*Lbt;
     L =[Ls,zeros(n1,n2);zeros(n2,n1),Lt];
     Lc =X*L*X';
    % update V
    VA = alpha*(U*U')+ (1+mu)*eye(size(U,1));
    VB = W'*Xs  + alpha*U*YY + mu*U - Z;
    V = VA\VB;
    V = (V./repmat(sqrt(sum(V.*V)),[size(V, 1) 1]));
    % update U
    UA = alpha*(V*V') + mu* eye(size(V,1));
    UB = alpha*V*YY + mu*V+Z;
    U = UA\UB;
    %Update P
     if iter==1
         P=P1;
     else
       A = X + F/mu;
       [U1,S1,V1] = svd(A*X'*W,'econ'); 
       P = U1*V1';
       clear A;
     end
    % update W
     A = X + F/mu; 
     WA = XX + beta*D + gamma*lg + mu*X*X'+lambda1*Lc;
     WB = Xs*V'+ mu*X*A'*P;
     W = WA\WB;
     v  = sqrt(sum(W.*W,2)+eps);
     D  = diag(1./(v));
     % update Z and theta
    Z = Z+mu*(V-U);
    F = F+mu*(X-P*W'*X);
    mu= min(mu_max, rho*mu);
    obj = 0.5*norm(V-W'*Xs, 'fro')^2 + alpha/2*norm(V'*U-YY, 'fro')^2  + beta*sum(v)+gamma*trace(W'*lg*W)+lambda1*trace(W'*Lc*W);
     regression= 0.5*norm(V-W'*Xs, 'fro')^2;
     recons = alpha/2*norm(V'*U-YY, 'fro')^2 ;
     orths = beta*sum(v);
     graph = gamma*trace(W'*lg*W);
     graph_local =lambda1*trace(W'*Lc*W);
     fobject=regression+recons+orths+graph+graph_local;
   % fprintf('Objective: %2.4f,regression: %2.4f,recons: %2.4f, orths: %2.4f,graph: %2.4f ,graph_local: %2.4f\n',fobject,regression,recons,orths,graph,graph_local); 
    end   
 end
function [y,acc] = myClassifier(Xs,Xt,Ys,Yt,p)
        X = [Xs,Xt];
        Z = p'*X;
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        D=EuDist2(Zt',Zs');
         [~,idx]=sort(D,2);
         y=Ys(idx(:,1),1);
         acc=length(find(y==Yt))/length(Yt);
         clear X Z Zs Zt D Xs Xt Ys Yt p
end

