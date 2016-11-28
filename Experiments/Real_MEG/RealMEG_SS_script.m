% This script is written in order to replicate the experimental results on
% real MEG data for SS-L1 ans SS-L2 approaches in paper entitled "Group-Level
% Spatio-Temporal Pattern Recovery in MEG Decoding using Multi-Task Joint
% Feature Learning". Please change "penalization" in line 20 to switch between
% SS-L1 and SS-L2.

% November 2016, Seyed Mostafa Kia (m.kia83@gmail.com)

clear all;
addpath(genpath('Path to MALSAR toolbox.'));
addpath('Path to Functions folder.');
datapath = 'Specify the real MEG data directory.';
savePath = 'Specify the save directory.';
%lambda = [0.001 0.1 1 5 10 25 50 100 200 300]; % for least squares loss
lambda = [0.001 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]; % for logistic loss
bootstrap_num = 50;
subjects_train = 1:16;
timeInterval = 76:325; % -200ms to 800ms

penalization = 'L1'; % Use 'L1' for l_1 regularization and 'L2' for l_2 regularization
loss = 'logistic'; % Use 'logistic' for logistic regression and 'least' for least squares loss

for subj = subjects_train
    filename = sprintf(datapath,subjects_train(subj));
    disp(strcat('Loading ',filename));
    data = load(filename);
    [trialNum,channelNum,timeNum] = size(data.X);
    data.y(data.y==0)=-1;
    data.X = data.X(:,:,timeInterval);
    X = reshape(data.X,[trialNum,channelNum*length(timeInterval)]);
    Y = single(data.y);
    X = mapstd(X')';
    n = size(X,1);
    A{subj} = mean(X(Y == 1,:)) - mean(X(Y == -1,:));
    clear data;
    
    % Training
    opts = [];
    opts.n = n;
    opts.penalization = penalization;
    opts.loss = loss;
    if strcmp(penalization,'L1')
        opts.alpha = 1;
    elseif strcmp(penalization,'L2')
        opts.alpha = 0;
    end
    
    for l = 1 : length(lambda)
        opts.lambda = lambda(l);
        [W,~,Y_table,acc] = OOB(X,Y,bootstrap_num,opts,0);
        ACC(subj,l) = mean(acc);
        [performance(subj,l)] = EPE(Y_table,Y);
        [interpretable(subj,l)] = interpretability(W,A{subj});
        zeta(subj,l) = zeta_phi(performance(subj,l).performance,interpretable(subj,l).interpretability,1,1,0.55);
        disp(strcat('Subject:',num2str(subj),',Lambda:',num2str(lambda(l)), ',Performance:',num2str(performance(subj,l).performance),...
            ',Interpretable:',num2str(interpretable(subj,l).interpretability),',Zeta:',num2str(zeta(subj,l))));
    end
    save(strcat(savePath,'ST_SS_Real_', loss, penalization,'_Results.mat'),'ACC','performance','zeta','interpretable','lambda','A');
end
