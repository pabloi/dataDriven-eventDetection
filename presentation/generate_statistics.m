% Load groundtruth
load('set1_1.mat')
goldL = squeeze(double(y(:,1,:)));
goldR = squeeze(double(y(:,2,:)));
[T,N] = size(goldL);

% % Load O'Connor
% load('oconnorResults/oconnor_Testset1_1.mat')
% out_events = y_out;
% outL = squeeze(double(out_events(:,1,:)));
% outR = squeeze(double(out_events(:,2,:)));

% % Load Miller
% load('millerResults/miller_Testset1_1_Trainset1_2.mat')
% out_events = predictedYtest;
% outL = squeeze(double(out_events(:,1,:)));
% outR = squeeze(double(out_events(:,2,:)));

% Load LSTM
load('lstmResults/conv_lstm_Testset1_1_Trainset1_1.h5327000train.mat')
out_events = predictedY;
outL = squeeze(double(out_events(:,1,:)));
outR = squeeze(double(out_events(:,2,:)));

agg_dev = [];
agg_mistake = [];
for i = 1:N
    [deviationL, mistakeL, d_listL] = ss_metric(goldL(:,i), outL(:,i));
    [deviationR, mistakeR, d_listR] = ss_metric(goldR(:,i), outR(:,i));

    agg_dev_tmp = [deviationL; deviationR];
    agg_dev = [agg_dev; agg_dev_tmp(:,2)];
    agg_mistake(i) = (mistakeL + mistakeR) / 2;
end

res = [mean(abs(agg_dev)), std(agg_dev), mean(agg_mistake), std(agg_mistake)];
% outpath = 'results_oconner_testset1_1.txt';
% outpath = 'results_miller_testset1_1.txt';
outpath = 'results_lstm_testset1_1.txt';
dlmwrite(outpath,res,'\t')
hist(agg_dev);
xlabel('dt')
% saveas(gcf, 'figures/hist_oconner_testset1_1.eps', 'epsc')
% saveas(gcf, 'figures/hist_miller_testset1_1.eps', 'epsc')
saveas(gcf, 'figures/hist_lstm_testset1_1.eps', 'epsc')
