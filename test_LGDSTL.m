clear;clc;
addpath libsvm-new;
result = [];
src_str = {'jaffe','jaffe','jaffe','CK','CK','CK','tefid','tefid','tefid','KDEF','KDEF','KDEF'};
tgt_str = {'CK','tefid','KDEF','jaffe','tefid','KDEF','CK','jaffe','KDEF','CK','tefid','jaffe'};
for i =1:length(tgt_str)
    src = src_str{i};
    tgt = tgt_str{i};
    fprintf(' %s vs %s ', src, tgt);
load(['tuxiang\' src '_LBP.mat']); 
    Xs = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2)); 
    Xs = zscore(Xs);
    Xs = normr(Xs)';
    Xs_label= trainIdx;
    clear Train_DAT;
    clear trainIdx;
load(['tuxiang\' tgt '_LBP.mat']); 
    Xt = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2)); 
    Xt = zscore(Xt);
    Xt = normr(Xt)';
    Xt_label= trainIdx;
    clear Train_DAT;
    clear trainIdx;
    class = max(Xs_label);
    n_per = 10;
    REPEAT = 1;
    Acc = [];
    param.dim = class;
    opts.mu = 0.001;
    opts.rho = 1.1;
    opts.Max_Iter = 10;
    param.lambda =0.001;
     alpha ={0.01};
    beta = {0.1};
    gamma ={0.01};
    lambda1 ={0.001};
for a =1:1
    for b = 1:1
       for c = 1:1
           for d =1:1
           param.alpha = alpha{a};
           param.beta = beta{b};
            param.gamma = gamma{c};
            param.lambda1 =lambda1{d};
  tic
  for repeat=1:REPEAT 
     [W] = LGDSTL(Xs,Xt,Xs_label,Xt_label, param, opts);
    fea_train = W'*Xs;
    fea_test  = W'*Xt;
    tr_n = fea_train./repmat(sqrt(sum(fea_train.*fea_train)), [size(fea_train, 1), 1]);
    tt_n = fea_test./repmat(sqrt(sum(fea_test.*fea_test)), [size(fea_test, 1), 1]);
     tmd = ['-s 0 -t 2 -g ' num2str(1e-3) ' -c ' num2str(1000)];
     model = svmtrain(Xs_label, tr_n', tmd);
     [predict, acc] = svmpredict(Xt_label, tt_n', model);
     acc = acc(1);    
     fprintf(' %2.2f \n',acc);
     Acc=[Acc,acc];
  toc
  end
        end
    end
    end
  
end
   result = [result;Acc];
    result1 =max(result); 
end
   result1