function [W,Y_table,acc] = OOB_MTL (data,target,bootstrap_num,opts)
% This functions performs OOB procedure on a given dataset and model and
% returns weight of OOB models, their predictions, and accuracy. See 
% https://arxiv.org/abs/1603.08704 for more information.
% Inputs:
%           data: input data organized as n*p samples where n is the number
%           samples and p is the number of features.
%           target: Label vector of data organized as n*1 vector. This
%           vector should contain 1 for positive and -1 for negative
%           classes.
%           bootstrap_num: number of bootstraps.
%           opts: parameters of the model.
%           parellel: 0 or 1. if one the code will run in parallel.
% Outputs:
%           W: is a 1*bootstrap_num cell that contains the weight vector of
%           the model in each bootstrap repitition.
%           Y_table: is a bootstrap_num*n matrix that contains the
%           prediction of the model for all test samples in each run of
%           bootstrap.
%           acc: is 1*bootstrap_num vector that contains prediction
%           accuracy of the model in each run of bootstrap.

% Developed by Seyed Mostafa Kia (m.kia83@gmail.com)

n = opts.n;
for i = 1 : length(n)
    Y_table{i} = nan(bootstrap_num,n(i));
    for f = 1 : bootstrap_num
        randInd{i}(f,:) = randi(n(i),[1,n(i)]);
    end
end
W = cell(1,bootstrap_num);
y_pred =cell(bootstrap_num,length(n));
for f = 1 : bootstrap_num
    X_tr = [];
    Y_tr = [];
    X_te = [];
    Y_te = [];
    for task = 1 : length(n)
        X_tr{task} = data{task}(randInd{task}(f,1:n(task)),:);
        Y_tr{task} = target{task}(randInd{task}(f,1:n(task)));
        X_te{task} = data{task}(setdiff(1:n(task),randInd{task}(f,1:n(task))),:);
        Y_te{task} = target{task}(setdiff(1:n(task),randInd{task}(f,1:n(task))));
    end
    if strcmp(opts.loss,'L21')
        [W{f}] = Least_L21(X_tr,Y_tr,opts.lambda,opts);
    elseif strcmp(opts.loss,'L1')
        [W{f}] = Least_Lasso(X_tr,Y_tr,opts.lambda,opts);
    elseif strcmp(opts.loss,'L2')
        opts.rho_L2 = opts.lambda;
        [W{f}] = Least_Lasso(X_tr,Y_tr,0,opts);
    end
    for task = 1 : length(n)
        y_pred{f,task}= X_te{task}*W{f}(:,task);
        y_pred{f,task}= sign(y_pred{f,task});
        y_pred{f,task}(y_pred{f,task}==0) = 1;
        acc(f,task) = mean(Y_te{task}==y_pred{f,task});
    end
    disp(strcat('Bootstrap:',num2str(f)));
end
for task = 1 : length(n)
    for f = 1 : bootstrap_num
        Y_table{task}(f,setdiff(1:n(task),randInd{task}(f,1:n(task)))) = y_pred{f,task};
    end
end