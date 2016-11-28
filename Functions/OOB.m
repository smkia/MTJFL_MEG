function [W,A,Y_table,acc] = OOB (data,target,bootstrap_num,opts,parallel)
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

[n] = size(data,1);
Y_table = nan(bootstrap_num,n);
W = cell(1,bootstrap_num);
A = cell(1,bootstrap_num);
y_pred =cell(1,bootstrap_num);
randInd = zeros(bootstrap_num,n);
for f = 1 : bootstrap_num
    randInd(f,:) = randi(n,[1,n]);
end
if parallel
    parfor f = 1 : bootstrap_num
        X_tr = [];
        Y_tr = [];
        X_te = [];
        Y_te = [];
        if strcmp(opts.loss,'least')
            X_tr = data(randInd(f,1:n),:);
            Y_tr = target(randInd(f,1:n));
            X_te = data(setdiff(1:n,randInd(f,:)),:);
            Y_te = target(setdiff(1:n,randInd(f,:)));
            [W{f}] = EN_LS(X_tr,Y_tr,opts);
            A{f} = mean(X_tr(Y_tr == 1,:)) - mean(X_tr(Y_tr == -1,:));
            y_pred{f}= sign(X_te*W{f});
            y_pred{f}(y_pred{f}==0) = 1;
            acc(f) = mean(Y_te==y_pred{f});
        elseif strcmp(opts.loss,'logistic')
            X_tr{1} = data(randInd(f,1:n),:);
            Y_tr{1} = target(randInd(f,1:n));
            X_te{1} = data(setdiff(1:n,randInd(f,:)),:);
            Y_te{1} = target(setdiff(1:n,randInd(f,:)));
            if strcmp(opts.penalization,'L1')
                [W{f}] = Logistic_Lasso(X_tr,Y_tr,opts.lambda,opts);
            elseif strcmp(opts.penalization,'L2')
                opts.rho_L2 = opts.lambda;
                [W{f}] = Logistic_Lasso(X_tr,Y_tr,0,opts);
            end
            A{f} = mean(X_tr{1}(Y_tr{1} == 1,:)) - mean(X_tr{1}(Y_tr{1} == -1,:));
            y_pred{f}= sign(X_te{1}*W{f});
            y_pred{f}(y_pred{f} == 0) = 1;
            acc(f) = mean(Y_te{1} == y_pred{f});
        end
        disp(strcat('Bootstrap:',num2str(f)));
    end
else
    for f = 1 : bootstrap_num
        X_tr = [];
        Y_tr = [];
        X_te = [];
        Y_te = [];
        if strcmp(opts.loss,'least')
            X_tr = data(randInd(f,1:n),:);
            Y_tr = target(randInd(f,1:n));
            X_te = data(setdiff(1:n,randInd(f,:)),:);
            Y_te = target(setdiff(1:n,randInd(f,:)));
            [W{f}] = EN_LS(X_tr,Y_tr,opts);
            A{f} = mean(X_tr(Y_tr == 1,:)) - mean(X_tr(Y_tr == -1,:));
            y_pred{f}= sign(X_te*W{f});
            y_pred{f}(y_pred{f}==0) = 1;
            acc(f) = mean(Y_te==y_pred{f});
        elseif strcmp(opts.loss,'logistic')
            X_tr{1} = data(randInd(f,1:n),:);
            Y_tr{1} = target(randInd(f,1:n));
            X_te{1} = data(setdiff(1:n,randInd(f,:)),:);
            Y_te{1} = target(setdiff(1:n,randInd(f,:)));
            if strcmp(opts.penalization,'L1')
                [W{f}] = Logistic_Lasso(X_tr,Y_tr,opts.lambda,opts);
            elseif strcmp(opts.penalization,'L2')
                opts.rho_L2 = opts.lambda;
                [W{f}] = Logistic_Lasso(X_tr,Y_tr,0,opts);
            end
            A{f} = mean(X_tr{1}(Y_tr{1} == 1,:)) - mean(X_tr{1}(Y_tr{1} == -1,:));
            y_pred{f}= sign(X_te{1}*W{f});
            y_pred{f}(y_pred{f} == 0) = 1;
            acc(f) = mean(Y_te{1} == y_pred{f});
        end
        disp(strcat('Bootstrap:',num2str(f)));
    end
end
for f = 1 : bootstrap_num
    Y_table(f,setdiff(1:n,randInd(f,:))) = y_pred{f};
end