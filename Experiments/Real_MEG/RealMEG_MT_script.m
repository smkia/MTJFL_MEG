% This script is written in order to replicate the experimental results on 
% real MEG data for MT-L21 approach in paper entitled "Group-Level 
% Spatio-Temporal Pattern Recovery in MEG Decoding using Multi-Task Joint Feature Learning". 

% November 2016, Seyed Mostafa Kia (m.kia83@gmail.com)

clear all;
addpath(genpath('Path to MALSAR toolbox.'));
addpath('Path to Functions folder.');
datapath = 'Specify the real MEG data directory.';
savePath = 'Specify the save directory.';

subjects_train = 1:16;
lambda = [0.001 0.1 1 5 10 25 50 100 200 300];
timeInterval = 76:325; % -200ms to 800ms
bootstrap_num = 50;

for subj = subjects_train 
    % Preparing data
    filename = sprintf(datapath,subjects_train(subj));
    disp(strcat('Loading ',filename));
    data = load(filename);
    [trialNum,channelNum,timeNum] = size(data.X);
    data.y(data.y==0)=-1;
    data.X = data.X(:,:,timeInterval);
    d{subj} = reshape(data.X,[trialNum,channelNum*length(timeInterval)]);
    target{subj} = single(data.y);
    d{subj} = mapstd(d{subj}')';
    [n(subj)] = size(d{subj},1);
    A{subj} = mean(d{subj}(target{subj}==1,:)) - mean(d{subj}(target{subj}==-1,:));
    clear data;
end

opts = [];
opts.tol = 10e-4;
opts.n = n;
opts.penalization = 'L21';
for l = 1 : length(lambda)
    opts.lambda = lambda(l);
    [W,Y_table,acc] = OOB_MTL(d,target,bootstrap_num,opts);
    ACC(l,:) = mean(acc);
    for subj = subjects_train
        [performance(subj,l)] = EPE(Y_table{subj},target{subj});
        temp_w = [];
        for i = 1 : bootstrap_num
            temp_w{i} = W{i}(:,subj);
        end
        [interpretable(subj,l)] = interpretability(temp_w,A{subj});
        zeta(subj,l) = zeta_phi(performance(subj,l).performance,interpretable(subj,l).interpretability,1,1,0.6);
        disp(strcat('Subject:',num2str(subj),',Lambda:',num2str(lambda(l)), ',Performance:',num2str(performance(subj,l).performance),...
        ',Interpretable:',num2str(interpretable(subj,l).interpretability),',Plausible:',num2str(zeta(subj,l))));
    end
    save(strcat(savePath,'MT_L21_Results.mat'),'ACC','performance','zeta','interpretable','lambda','A');
end