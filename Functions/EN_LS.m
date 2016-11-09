function [W] = EN_LS(X, Y, opts)
% This function implements elasticnet with least squares loss for
% classification.
% Inputs:   
%           X:  n*p input matrix where n is number of samples and p is the
%           number of features.
%           Y: n*1 vector of labels.
%           opts.alpha: a number between 0 and 1 that defines the weight of L1
%               and L2 penalization. 0 means just L2 penalization and 1 is
%               Lasso (Default = 0).
%           opts.lambda: the regularization parameter (Default = 0).   
%           opts.tol: the tolerance in optimization (Default = 10e-4).  
%           opts.maxIter: Maximum iterations in optimization (Default = 1000).  
%           opts.tFlag: Termination condition
%                           0 => change of absolute function value:
%                             abs( funcVal(i)- funcVal(i-1) ) <= .tol
%                           1 (Default) => change of relative function value:
%                             abs( funcVal(i)- funcVal(i-1) )
%                              <= .tol * funcVal(i-1)
%                           2 => absolute function value:
%                             funcVal(end)<= .tol
%                           3 => Run the code for .maxIter iterations
% Outputs:
%           W: p*1 vector weight of linear classifier.

% See http://www.public.asu.edu/~jye02/Software/MALSAR/ for the original
% implementation.

X = X';

% initialize options.
if isfield(opts, 'tol')
    if (opts.tol <eps * 100)
        opts.tol = eps * 100;
    end
else
    opts.tol = 10e-4;
end
if isfield(opts, 'maxIter')
    if (opts.maxIter<1)
        opts.maxIter = 1000;
    end
else
    opts.maxIter = 1000;
end
if isfield(opts,'tFlag')
    if opts.tFlag<0
        opts.tFlag=0;
    elseif opts.tFlag>3
        opts.tFlag=3;
    else
        opts.tFlag=floor(opts.tFlag);
    end
else
    opts.tFlag = 1;
end

if ~isfield(opts,'lambda')
    opts.lambda = 0;
end
if ~isfield(opts,'alpha')
    opts.alpha = 0;
end

rho1 = opts.lambda * opts.alpha;
rho_L2 = opts.lambda * (1 - opts.alpha);

dimension = size(X, 1);
funcVal = [];

% initialize a starting point
W0 = zeros(dimension, 1);

bFlag=0; % this flag tests whether the gradient step only changes a little

Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;


iter = 0;
gamma = 1;
gamma_inc = 2;
XY= X * Y;
while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval  (Ws);
    
    while true
        [Wzp, l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho1 / gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp)^2;
        Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp)^2;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + rho1 * l1c_wzp);
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;


% private functions

    function [z, l1_comp_val] = l1_projection (v, beta)
        % this projection calculates
        % argmin_z = \|z-v\|_2^2 + beta \|z\|_1
        % z: solution
        % l1_comp_val: value of l1 component (\|z\|_1)
        z = zeros(size(v));
        vp = v - beta/2;
        z (v> beta/2)  = vp(v> beta/2);
        vn = v + beta/2;
        z (v< -beta/2) = vn(v< -beta/2);
        l1_comp_val = sum(sum(abs(z)));
    end

    function [grad_W] = gradVal_eval(W)
        XW = X' * W;
        XXW = X * XW;
        grad_W = XXW - XY;
        grad_W = grad_W + rho_L2 * 2 * W;
    end

    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        funcVal = funcVal + 0.5 * norm (Y - X' * W)^2 + rho_L2 * norm(W)^2;
    end
end