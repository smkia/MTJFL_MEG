function [res] = EPE(Y_table,Y)
% This function receives OOB Y_table and the actual targest and computes
% EPE of the model. See https://arxiv.org/abs/1603.08704 for more
% information.
% Inputs:   
%           Y_table: is a B*n matirx of predicted labels for then sample via OOB procedure.
%           Y: n*1 vector of actual labels.
% Outputs:
%           res: a structure that contains bias, variance,
%           unbiased-variance, biased-variance, EPE and the performance of
%           the model.

% Developed by Seyed Mostafa Kia (m.kia83@gmail.com)

bootstrap_num = size(Y_table,1);
main_prediction = sign(nanmean(Y_table));
bias = (main_prediction~=Y');
variance = nanmean((Y_table~=repmat(main_prediction,bootstrap_num,1))+ 0*Y_table);
bVariance = nanmean((Y_table(:,bias)~=repmat(main_prediction(bias),bootstrap_num,1))+ 0*Y_table(:,bias));
uVariance = nanmean((Y_table(:,~bias)~=repmat(main_prediction(~bias),bootstrap_num,1))+ 0*Y_table(:,~bias));
res.UV = nanmean([uVariance zeros(1,length(bVariance))]);
res.BV = nanmean([bVariance zeros(1,length(uVariance))]);
res.BS = nanmean(double(bias));
res.VR = nanmean(variance);
res.EPE = res.BS + res.UV - res.BV;
res.performance = 1 - res.EPE;
