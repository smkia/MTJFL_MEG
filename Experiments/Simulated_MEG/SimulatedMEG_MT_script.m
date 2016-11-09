% This script is written in order to replicate the simulation results for MT-L21
% approach in paper entitled "Group-Level Spatio-Temporal Pattern Recovery in MEG 
% Decoding using Multi-Task Joint Feature Learning". 

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
cfgSimulation.trialNumPerClass = 250; % 250
cfgSimulation.trialLength = 100;
cfgSimulation.channelType = 'MEGMAG';
cfgSimulation.jitter = 1;
cfgSimulation.bpfilter = [0.3,45];
cfgSimulation.freq = freq;
cfgSimulation.effectTime = effectTime;
trialNum = 2 * cfgSimulation.trialNumPerClass;

for iter = 1 : iterNum
    % Preparing data
    for s = 1 : subNum
        cfgSimulation.sourceMom = sourceMom(s,:);
        [raw,raw_GT] = MEG_simulation(cfgSimulation);
        [channelNum,timeNum] = size(raw{1}.trial{1});
        X{s} = zeros(trialNum,channelNum*timeNum);
        Y{s} = zeros(trialNum,1);
        for i = 1 : cfgSimulation.trialNumPerClass
            X{s}(i,:) = reshape(raw{1}.trial{i},[1,channelNum*timeNum]);
            Y{s}(i) = 1;
        end
        for i = 1 : cfgSimulation.trialNumPerClass
            X{s}(cfgSimulation.trialNumPerClass+i,:) = reshape(raw{2}.trial{i},[1,channelNum*timeNum]);
            Y{s}(cfgSimulation.trialNumPerClass+i) = -1;
        end
        idx = randperm(trialNum);
        X{s} = X{s}(idx,:);
        Y{s} = Y{s}(idx,:);
        X{s} = single(mapstd(X{s}')');
        Y{s} = single(Y{s});
        A{s,iter} = mean(X{s}(Y{s}==1,:)) - mean(X{s}(Y{s}==-1,:));
        A{s,iter} = A{s,iter}/norm(A{s,iter});
        GT{s} = reshape(raw_GT.trial{1},[channelNum*timeNum,1]);
        GT{s} = GT{s}/norm(GT{s});
        n(s) = trialNum;
    end
    
    % Training
    opts = [];
    opts.tol = 10e-4;
    opts.n = n;
    opts.loss = 'L21';
    for l = 1 : length(lambda)
        opts.lambda = lambda(l);
        [W,Y_table,acc] = OOB_MTL(X,Y,bootstrap_num,opts);
        ACC(iter,l,:) = mean(acc);
        for subj = 1 : length(X)
            [performance(iter,subj,l)] = EPE(Y_table{subj},Y{subj});
            temp_w = [];
            for i = 1 : bootstrap_num
                temp_w{i} = W{i}(:,subj);
            end
            [interpretable(iter,subj,l)] = interpretability(temp_w,A{subj,iter});
            zeta(iter,subj,l) = zeta_phi(performance(iter,subj,l).performance,interpretable(iter,subj,l).interpretability,1,1,0.55);
            disp(strcat('Iter:',num2str(iter),',Subject:',num2str(subj),',Lambda:',num2str(lambda(l)), ',Performance:',num2str(performance(iter,subj,l).performance),...
                ',Interpretable:',num2str(interpretable(iter,subj,l).interpretability),',Plausible:',num2str(zeta(iter,subj,l))));
        end
    end
    save(strcat(savePath,'SimulatedMEG_MT_',opts.loss,'_Results.mat'),'ACC','performance','zeta','interpretable','lambda'...
                ,'GT','A');
end