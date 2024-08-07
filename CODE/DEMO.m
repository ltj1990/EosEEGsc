close all;
clear all;
clc;
addpath('measures');
addpath('EosEEGsc');

load ('./data/IV_2a_s1.mat');
unregu_data = Xt';
L_true = Yt';
num_cluster = size(unique(L_true),2); 
feature=z_regularization(unregu_data);
[A_x, A_h, runtime1] = PreProcess(feature, num_cluster);
ResBest = zeros(1, 8);
ResStd = zeros(1, 8);
% parameters setting
% r1 = 0 : 0.1 : 10;
r1 = [0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 5.0 10.0];
acc = zeros(length(r1), 1);
nmi = zeros(length(r1), 1);
purity = zeros(length(r1), 1);
idx = 1;
Result=[];
for r1Index = 1 : length(r1)
    r1Temp = r1(r1Index);
    fprintf('Please wait a few minutes\n');
    tic;
    [H_star, F, A_star, Obj] = EosEEGsc(A_x, A_h, num_cluster, r1Temp);
    [~ , label] = max(F, [], 2);
    toc;
    [RI,fscore,kappa,NMI] = RI_F1_kapa_nmi(L_true,label,num_cluster);
    Result=[Result; RI NMI fscore kappa r1Temp];
end
Result

