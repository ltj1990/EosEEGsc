clc
clear
load('II_Ia_Ib_data.mat');
data = Data;
L_true=data(:,1);
[mD nD]=size(data);
unregu_data=data(:,2:end);
data =  reshape(unregu_data,[468,6,896]);
fea = permute(data,[2 3 1]);
Cn = centroid_align(fea,'euclid');
Xt=logmap(Cn,'MI'); 
Yt = L_true;