function [res] = interpretability(W,A)
% interpretability function recieves the Theta^cERF or Theta^* plus the
% oob's weights of the linear model as input and computes the reproducibility,
% repreentativeness, and the interpretability of the model. See https://arxiv.org/abs/1603.08704 for more
% information.
% Inputs: 
%              W: is a 1*B cell array that contains B weight vecotors
%              resulting from B bootstrap in OOB.
%              A: is p*1 vectot that  represents theta^IBDS or Theta^*
% Ouputs:
%              res: is a struct that contains the computed
%              representativeness, reproducibility, and interpretability.

% Developed by Seyed Mostafa Kia (m.kia83@gmail.com)

bsn = length(W);
w_temp = zeros(length(W{1}),bsn);
s_temp = zeros(bsn,1);
i_temp = zeros(bsn,1);
for i = 1 : bsn
    w_temp(:,i) = W{i}(:)/norm(W{i}(:));  
end
main_map = mean(w_temp,2)/norm(mean(w_temp,2));
A = A(:);
res.representativeness = abs(1 - pdist([main_map';A'],'cosine'));
if isnan(res.representativeness)
    res.representativeness=0;
end
for i = 1 : bsn
    s_temp(i) = abs(1 - pdist([main_map';w_temp(:,i)'],'cosine'));
    i_temp(i) = abs(1 - pdist([A';w_temp(:,i)'],'cosine'));
end
s_temp(isnan(s_temp))=0;
i_temp(isnan(i_temp))=0;
res.reproducibility = mean(s_temp);
res.interpretability = mean(i_temp);