% This script is written in order to replicate the simulation results for SS-L1
% and SS-L2 approaches in paper entitled "Group-Level Spatio-Temporal Pattern Recovery in MEG
% Decoding using Multi-Task Joint Feature Learning". Please change
% "penalization" in line 33 to switch between SS-L1 and SS-L2.

% November 2016, Seyed Mostafa Kia (m.kia83@gmail.com)

clear all;
addpath(genpath('Path to MALSAR toolbox.'));
addpath(genpath('Path to Fieldtrip toolbox.'))
addpath('Path to Functions folder.');
savePath = 'Specify the save directory.';
lambda = [0.001 0.1 1 5 10 25 50 100 200 300];
bootstrap_num = 50; 
iterNum = 10;  
subNum = 7; 
effectTime = [45,75];
freq = [3,5];
sourcePos = [-4.7 -3.7 5.3]; 
sourceMom = [1,0,0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 1 0 1; 1 1 1];

cfgSimulation.sourcePos = sourcePos;
cfgSimulation.samplingRate = 300;
cfgSimulation.trialNumPerClass = 250; 
cfgSimulation.trialLength = 100;
cfgSimulation.channelType = 'MEGMAG';
cfgSimulation.jitter = 1;
cfgSimulation.bpfilter = [0.3,45];
cfgSimulation.freq = freq;
cfgSimulation.effectTime = effectTime;
trialNum = 2 * cfgSimulation.trialNumPerClass;

penalization = 'L1'; % Use 'L1' for l_1 regularization and 'L2' for l_2 regularization

for iter = 1 : iterNum
    % Preparing data
    d = [];
    target = [];
    for s = 1 : subNum
        cfgSimulation.sourceMom = sourceMom(s,:);
        [raw,raw_GT] = MEG_simulation(cfgSimulation);
        [channelNum,timeNum] = size(raw{1}.trial{1});
        X = zeros(trialNum,channelNum*timeNum);
        Y = zeros(trialNum,1);
        for i = 1 : cfgSimulation.trialNumPerClass
            X(i,:) = reshape(raw{1}.trial{i},[1,channelNum*timeNum]);
            Y(i) = 1;
        end
        for i = 1 : cfgSimulation.trialNumPerClass
            X(cfgSimulation.trialNumPerClass+i,:) = reshape(raw{2}.trial{i},[1,channelNum*timeNum]);
            Y(cfgSimulation.trialNumPerClass+i) = -1;
        end
        idx = randperm(trialNum);
        X = single(X(idx,:));
        Y = single(Y(idx,:));
        X = mapstd(X')';
        [n] = size(X,1);
        A{iter,s} = mean(X(Y==1,:)) - mean(X(Y==-1,:));
        A{iter,s} = A{iter,s}/norm(A{iter,s});
        
        % Training
        opts = [];
        opts.tol = 10e-4;
        opts.n = n;
        if strcmp(penalization,'L1')
            opts.alpha = 1;
        elseif strcmp(penalization,'L2')
            opts.alpha = 0;
        end
        for l = 1 : length(lambda)
            opts.lambda = lambda(l);
            [W,~,Y_table,acc] = OOB(X,Y,bootstrap_num,opts,0);
            ACC(iter,s,l) = mean(acc);
            [performance(iter,s,l)] = EPE(Y_table,Y);
            [interpretable(iter,s,l)] = interpretability(W,A{iter,s});
            zeta(iter,s,l) = zeta_phi(performance(iter,s,l).performance,interpretable(iter,s,l).interpretability,1,1,0.55);
            disp(strcat('Iter:',num2str(iter),',Subject:',num2str(s),',Lambda:',num2str(lambda(l)), ',Performance:',num2str(performance(iter,l).performance),...
                ',Interpretable:',num2str(interpretable(iter,l).interpretability),',Zeta:',num2str(zeta(iter,l))));
        end
        save(strcat(savePath,'ST_SS_Simulation_',penalization,'_Results.mat'),'ACC','performance','zeta','interpretable','lambda','A');
    end
end