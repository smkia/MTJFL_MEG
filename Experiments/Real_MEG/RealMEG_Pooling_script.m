% This script is written in order to replicate the experimental results on
% real MEG data for Pooling-L1 ans Pooling-L2 approaches in paper entitled "Group-Level
% Spatio-Temporal Pattern Recovery in MEG Decoding using Multi-Task Joint
% Feature Learning". Please change "penalization" in line 21 to switch between
% Pooling-L1 and Pooling-L2.

% November 2016, Seyed Mostafa Kia (m.kia83@gmail.com)

clear all;
addpath(genpath('Path to MALSAR toolbox.'));
addpath('Path to Functions folder.');
datapath = 'Specify the real MEG data directory.';
savePath = 'Specify the save directory.';

subjects_train = 1:16;
%lambda = [0.001 0.1 1 5 10 25 50 100 200 300]; % for least squares loss
lambda = [0.001 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]; % for logistic loss
timeInterval = 76:325; % -200ms to 800ms
bootstrap_num = 50;

penalization = 'L1'; % Use 'L1' for l_1 regularization and 'L2' for l_2 regularization
loss = 'logistic'; % Use 'logistic' for logistic regression and 'least' for least squares loss

d = [];
target = [];
for subj = subjects_train
    % Preparing data
    filename = sprintf(datapath,subjects_train(subj));
    disp(strcat('Loading ',filename));
    data = load(filename);
    [trialNum,channelNum,timeNum] = size(data.X);
    data.y(data.y==0)=-1;
    data.X = data.X(:,:,timeInterval);
    d = [d ; reshape(data.X,[trialNum,channelNum*length(timeInterval)])];
    target = [target;single(data.y)];
    clear data;
end

d = mapstd(d')';
[n] = size(d,1);
A = mean(d(target==1,:)) - mean(d(target==-1,:));

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
    [W,~,Y_table,acc] = OOB(d,target,bootstrap_num,opts,0);
    ACC(l,:) = mean(acc);
    [performance(l)] = EPE(Y_table,target);
    [interpretable(l)] = interpretability(W,A);
    zeta(l) = zeta_phi(performance(l).performance,interpretable(l).interpretability,1,1,0.6);
    disp(strcat('Lambda:',num2str(lambda(l)), ',Performance:',num2str(performance(subj,l).performance),...
        ',Interpretable:',num2str(interpretable(subj,l).interpretability),',Zeta:',num2str(zeta(subj,l))));
    save(strcat(savePath,'ST_Pooling_', loss, penalization,'_Results.mat'),'ACC','performance','zeta','interpretable','lambda','A');
end