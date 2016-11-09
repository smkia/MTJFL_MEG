% This function is direct implementation for the algorithm 1 in Appendix A
% of paper entitled "Group-Level Spatio-Temporal Pattern Recovery in MEG 
% Decoding using Multi-Task Joint Feature Learning".

% November 2016, Seyed Mostafa Kia (m.kia83@gmail.com)

function [W_MT,iter] = MT_L21(X, Y, lambda, tol, maxIter)
task_num  = length (X);
dimension = size(X{1}, 2);

W0 = zeros(dimension,task_num);
W1 = zeros(dimension,task_num);
V0 = 0;
V1 = inf;
alpha0 = 0;
alpha1 = 1;
gamma = 1;
iter = 1;
while abs(V1 + nonsmooth_eval(W1,lambda) - V0 - nonsmooth_eval(W0,lambda)) > tol * (V0 + nonsmooth_eval(W0,lambda)) && iter<=maxIter
    S = W1 + ((alpha0-1)/alpha1) * (W1 - W0);
    G = zeros(size(S));
    F = 0;
    for t = 1 : task_num
        G(:,t) = X{t}'*(X{t} * S(:,t) - Y{t});
        F = F + 0.5 * norm(Y{t} - X{t} * S(:, t))^2;
    end
    delta = ones(size(W1));
    while norm(delta,'fro')^2 > 1e-20
        U = S - G/gamma;
        eta = lambda/gamma;
        L = repmat(max(0, 1 - eta./sqrt(sum(U.^2,2))),1,size(U,2)).*U;
        delta = L - S;
        V0 = V1;
        V1 = 0;
        for t = 1 : task_num
            V1 = V1 + 0.5 * norm(Y{t} - X{t} * L(:, t))^2;
        end
        if V1 <= F + sum(sum(delta .* G)) + gamma/2 * norm(delta, 'fro')^2
            break;
        else
            gamma = gamma * 2;
        end
    end
    W0 = W1;
    W1 = L;
    
    
    alpha0 = alpha1;
    alpha1 = 0.5 * (1 + (1+ 4 * alpha1^2)^0.5);
    iter = iter +1;
end
W_MT = W1;


    function [non_smooth_value] = nonsmooth_eval(W, rho_1)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_1 * norm(w, 2);
        end
    end
end