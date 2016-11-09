function [criterion] = zeta_phi(per,int,omega1,omega2,kappa)
% This function receives computes the multi-objective criterion zeta for the model
% selection by combing the generalization perfromance and interpretability
% of models. See https://arxiv.org/abs/1603.08704 for more information.
% Inputs:   
%           per: is the generalization performance of the model.
%           int: is the interpretability of the model.
%           omega1: is the weight of the generalization performance.
%           omega2: is the weight of the interpretability.
%           kappa: is the threshold on the performance.
% Output:
%           criterion: the computed multi-objective criterion zeta

% Developed by Seyed Mostafa Kia (m.kia83@gmail.com)

if per >= kappa
    criterion = (omega1*per + omega2*int)/(omega1+omega2); 
else
    criterion = 0;
end