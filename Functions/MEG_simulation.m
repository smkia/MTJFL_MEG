function [raw,raw_GT] = MEG_simulation(cfg)
% This function simulates MEG data based on the configurations in cfg. The
% fieldtrip function ft_dipolesimulation is used for dipole simulation.
% Inputs:   
%           cfg.sourcePos: Dipole position in source space, e.g., [-4.7 -3.7 5.3];
%           cfg.sourceMom: Dipole direction (x,y,z) in source space, e.g., [1,1,0];
%           cfg.samplingRate: the sampling rate.
%           cfg.trialNumPerClass: number of trials per class.
%           cfg.trialLength: number of time-points in each trial.
%           cfg.channelType: MEG channels to be simulated 'MEGMAG' for
%           magnetometers, 'MEGGRAD' for gradiometers, and 'all' for full simulation.
%           cfg.freq: a 1 by 2 vector that contains the frequency of peaks
%           in target class, e.g., [3,5].
%           cfg.jitter: Defines standard deviation of jitter of peaks. 
%           cfg.bpfilter: a 1 by 2 vector that defines the cut-off
%           frequency for bandpass filter, e.g., [0.3,45].
% Outputs:
%           raw: a 1 by 2 cell of simulated samples of two classes. raw{1}
%           contains the samples of positive class and raw{2} is for
%           negative class.
%           raw_GT: represents the ground-truth effect and contains the true effect in the positive (target) class
%           without noise contamination.

% Developed by Seyed Mostafa Kia (m.kia83@gmail.com).

% Loading the required data for the simulation.
grad = load('neuromag306grad');
for i = 1:length(grad.label); grad.label{i}(strfind(grad.label{i}, ' ')) = '';end;
load('MRIinfo.mat');
load('MRI_aligned.mat');
load('meanpower.mat')


sourcePos = cfg.sourcePos;
sourceMom = cfg.sourceMom;
samplingRate = cfg.samplingRate;
trialNumPerClass = cfg.trialNumPerClass;
trialLength = cfg.trialLength;
channelType = cfg.channelType;
freq = cfg.freq;
jitter = cfg.jitter;
bpfilter = cfg.bpfilter;
effectTime = cfg.effectTime;

% The effect
for i = 1 : trialNumPerClass
    s{i} = meanpower(freq(1)) / meanpower(1) * peak (trialLength, 1, samplingRate, freq(1), effectTime(1), jitter) ...
        + noise(trialLength, 1, samplingRate) - meanpower(freq(2))/ meanpower(1) * peak (trialLength, 1, samplingRate,  freq(2), effectTime(2), jitter); 
end
effect{1} = s;
for i = 1 : trialNumPerClass
    s{i} = noise(trialLength, 1, samplingRate);
end
effect{2} = s;

% dipole simulation
cfg                     = [];
cfg.channel             = channelType;
cfg.grad                = grad;
cfg.headmodel           = hdm;
cfg.ntrials             = trialNumPerClass;
cfg.triallength         = trialLength/samplingRate; %sec
cfg.fsample             = samplingRate;
cfg.relnoise            = 1;

cfg.dip.pos             = sourcePos; %cm
cfg.dip.mom             = sourceMom';
cfg.dip.signal          = effect{1};
raw{1}                  = ft_dipolesimulation(cfg);

cfg.dip.pos             = sourcePos; %cm
cfg.dip.mom             = sourceMom';
cfg.dip.signal          = effect{2};
raw{2}                  = ft_dipolesimulation(cfg);

cfg.ntrials             = 1;
cfg.relnoise            = 0;
cfg.absnoise            = 0;
cfg.dip.pos             = sourcePos; %cm
cfg.dip.mom             = sourceMom';
cfg.dip.signal          =  meanpower(freq(1)) / meanpower(1) * peak (trialLength, 1, samplingRate, freq(1), effectTime(1), 0) ...
                            - meanpower(freq(2))/ meanpower(1) * peak (trialLength, 1, samplingRate, freq(2), effectTime(2) , 0);
raw_GT                  = ft_dipolesimulation(cfg);


% Preprocessing
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = bpfilter;
raw{1} = ft_preprocessing(cfg,raw{1});
raw{2} = ft_preprocessing(cfg,raw{2});
