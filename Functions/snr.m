function s = snr(noise, signal)

mean_original = mean(signal(:));
tmp           = signal - mean_original;
var_original  = sum(tmp(:).^2);

mean_noise = mean(noise(:));
tmp        = noise - mean_noise;
var_noise  = sum(tmp(:).^2);

if var_noise == 0
    s = 999.99; %% INF. clean image
else
    s = 10 * log10(var_original / var_noise);
end
