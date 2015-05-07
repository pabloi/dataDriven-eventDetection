%% Split dataArray_master.mat into train and test sets
clearvars
load ../data/dataArrays_master.mat

%% Separate indexes
actualIdxs=cellfun(@(x) ~isempty(x),names);
strokeIdxs=cellfun(@(x) ~isempty(x),regexp(names(actualIdxs),'^P','once'));
healthyIdxs=~strokeIdxs;

%% 0: just one trial, split in half, for testing purposes
X=motionArray(:,:,1);
y=uint8(stanceArray(:,:,1));
N=size(X,1);
X1=X(1:floor(N/2),:,:);
X2=X(floor(N/2)+1:N,:,:);
y1=y(1:floor(N/2),:,:);
y2=y(floor(N/2)+1:N,:,:);
markerLabels=actualMarkers{1};

X=X1;y=y1;
save ../data/set0_1.mat X y markerLabels
X=X2;y=y2;
save ../data/set0_2.mat X y markerLabels

% M=size(X1);
% O=size(y1);
% M2=size(X1);
% O2=size(y1);
% for i=1
%     h5create('../data/set0_1.h5',['/' num2str(i) '/1/X'],M(2:-1:1))
%     h5create('../data/set0_1.h5',['/' num2str(i) '/1/y'],O(2:-1:1))
%     h5write('../data/set0_1.h5',['/' num2str(i) '/1/X'],X1(:,:,i)')
%     h5write('../data/set0_1.h5',['/' num2str(i) '/1/y'],y1(:,:,i)')
%     h5create('../data/set0_2.h5',['/' num2str(i) '/1/X'],M2(2:-1:1))
%     h5create('../data/set0_2.h5',['/' num2str(i) '/1/y'],O2(2:-1:1))
%     h5write('../data/set0_2.h5',['/' num2str(i) '/1/X'],X2(:,:,i)')
%     h5write('../data/set0_2.h5',['/' num2str(i) '/1/y'],y2(:,:,i)')
% end

%% 1: for all HEALTHY subjects, randomly separate the sequences in half (along time) and put in two sets
X=motionArray(:,:,find(healthyIdxs));
y=uint8(stanceArray(:,:,find(healthyIdxs)));
N=size(X,1);
X1=X(1:floor(N/2),:,:);
X2=X(floor(N/2)+1:N,:,:);
y1=y(1:floor(N/2),:,:);
y2=y(floor(N/2)+1:N,:,:);
markerLabels=actualMarkers{1};

X=X1;y=y1;
save ../data/set1_1.mat X y markerLabels
X=X2;y=y2;
save ../data/set1_2.mat X y markerLabels
% 
% M=size(X1);
% O=size(y1);
% M2=size(X1);
% O2=size(y1);
% for i=1:M(3)
%     h5create('../data/set1_1.h5',['/' num2str(i) '/1/X'],M(2:-1:1))
%     h5create('../data/set1_1.h5',['/' num2str(i) '/1/y'],O(2:-1:1))
%     h5write('../data/set1_1.h5',['/' num2str(i) '/1/X'],X1(:,:,i)')
%     h5write('../data/set1_1.h5',['/' num2str(i) '/1/y'],y1(:,:,i)')
%     h5create('../data/set1_2.h5',['/' num2str(i) '/1/X'],M2(2:-1:1))
%     h5create('../data/set1_2.h5',['/' num2str(i) '/1/y'],O2(2:-1:1))
%     h5write('../data/set1_2.h5',['/' num2str(i) '/1/X'],X2(:,:,i)')
%     h5write('../data/set1_2.h5',['/' num2str(i) '/1/y'],y2(:,:,i)')
% end

%% 1 ALT: for all HEALTHY subjects, randomly separate the sequences in half (along time) and put in two sets. Standardized
X=motionArray(:,:,find(healthyIdxs));
X=bsxfun(@rdivide,bsxfun(@minus,X,mean(X,1)),std(X,[],1));
y=uint8(stanceArray(:,:,find(healthyIdxs)));
N=size(X,1);
X1=X(1:floor(N/2),:,:);
X2=X(floor(N/2)+1:N,:,:);
y1=y(1:floor(N/2),:,:);
y2=y(floor(N/2)+1:N,:,:);
markerLabels=actualMarkers{1};

X=X1;y=y1;
save ../data/set1_1s.mat X y markerLabels
X=X2;y=y2;
save ../data/set1_2s.mat X y markerLabels

M=size(X1);
O=size(y1);
M2=size(X1);
O2=size(y1);
for i=1:M(3)
    h5create('../data/set1_1s.h5',['/' num2str(i) '/1/X'],M(2:-1:1))
    h5create('../data/set1_1s.h5',['/' num2str(i) '/1/y'],O(2:-1:1))
    h5write('../data/set1_1s.h5',['/' num2str(i) '/1/X'],X1(:,:,i)')
    h5write('../data/set1_1s.h5',['/' num2str(i) '/1/y'],y1(:,:,i)')
    h5create('../data/set1_2s.h5',['/' num2str(i) '/1/X'],M2(2:-1:1))
    h5create('../data/set1_2s.h5',['/' num2str(i) '/1/y'],O2(2:-1:1))
    h5write('../data/set1_2s.h5',['/' num2str(i) '/1/X'],X2(:,:,i)')
    h5write('../data/set1_2s.h5',['/' num2str(i) '/1/y'],y2(:,:,i)')
end

%% 2: for all HEALTHY subjects, randomly separate the subjects in half, to study generalization power
X=motionArray(:,:,find(healthyIdxs));
y=stanceArray(:,:,find(healthyIdxs));
N=size(X,3);
X1=X(:,:,1:round(N/2));
X2=X(:,:,round(N/2)+1:N);
y1=y(:,:,1:round(N/2));
y2=y(:,:,round(N/2)+1:N);
markerLabels=actualMarkers{1};

X=X1;y=y1;
save ../data/set2_1.mat X y markerLabels
X=X2;y=y2;
save ../data/set2_2.mat X y markerLabels
%% 3: for all subjects, repeat 1
X=motionArray(:,:,find(actualIdxs));
y=stanceArray(:,:,find(actualIdxs));
N=size(X,1);
X1=X(1:round(N/2),:,:);
X2=X(round(N/2)+1:N,:,:);
y1=y(1:round(N/2),:,:);
y2=y(round(N/2)+1:N,:,:);
markerLabels=actualMarkers{1};

X=X1;y=y1;
save ../data/set3_1.mat X y markerLabels
X=X2;y=y2;
save ../data/set3_2.mat X y markerLabels

%% 4: for all subjects, repeat 2
X=motionArray(:,:,find(actualIdxs));
y=stanceArray(:,:,find(actualIdxs));
N=size(X,3);
X1=X(:,:,1:round(N/2));
X2=X(:,:,round(N/2)+1:N);
y1=y(:,:,1:round(N/2));
y2=y(:,:,round(N/2)+1:N);
markerLabels=actualMarkers{1};

X=X1;y=y1;
save ../data/set4_1.mat X y markerLabels
X=X2;y=y2;
save ../data/set4_2.mat X y markerLabels

%% 5: use all healthy subjects as test, and all strokes as train. This should be crap.

%% 6: 