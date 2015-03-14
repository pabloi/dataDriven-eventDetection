function [features, labels] = getNNFeatures(motionData,labels,fs,fcut)

%As described in Miller, we take just the heel and toe marker for each
%subject. Then low-pass filter, and compute both velocity and acceleration.

%% Step 1: find the relevant markers
markerList={'LTOEx','LTOEy','LTOEz','LHEEx','LHEEy','LHEEz','RTOEx','RTOEy','RTOEz','RHEEx','RHEEy','RHEEz'};
N=size(motionData,1);
M=length(markerList);
data=NaN(N,M);
for i=1:length(markerList)
    data(:,i)=motionData(:,strcmp(markerList{i},labels));
end

%% Step 2: low-pass filter @10Hz
Wn=2*fcut/fs;
lowPassFilter=design(fdesign.lowpass('Fp,Fst,Ap,Ast',Wn,1.2*Wn,3,10),'butter');
data=filtfilthd(lowPassFilter,data);  %Ext function

%% Step 3: compute foot angle to horizontal plane 
%Note: assuming z=constant is the horizontal plane.
angleL=atan2(sqrt((data(:,2)-data(:,5)).^2+(data(:,1)-data(:,4)).^2),data(:,3)-data(:,6));
angleR=atan2(sqrt((data(:,8)-data(:,11)).^2+(data(:,7)-data(:,10)).^2),data(:,9)-data(:,12));
data=[data,angleL,angleR]; %Adding two new columns

%% Step 4: compute derivatives
dataV=.5*([zeros(1,M+2);fs*diff(data)] + [fs*diff(data);zeros(1,M+2)]); %So that it is centered
dataA=[zeros(1,M+2);fs^2*diff(diff(data));zeros(1,M+2)];

%% Step 5: Put everything together
labels=[markerList, 'angleL', 'angleR'];
labels=[labels, strcat('V',labels), strcat('A',labels)];
features=[data, dataV, dataA];

end

