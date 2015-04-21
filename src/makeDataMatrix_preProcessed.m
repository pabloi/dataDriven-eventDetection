subs=[20,21,23,35,36,37,40,41]; %Ignoring subject 34 because for some reason the TM base trial is not found

%Initialize
clear motionData motionTime metaData gaitEvents gaitTime
markersToBeUsed=['TOE';'ANK';'HEE';'KNE';'HIP';'TIB';'THI';'PSI';'ASI'];
markersToBeUsed=[['L' *ones(9,1),markersToBeUsed];[ 'R'*ones(9,1),markersToBeUsed]];
markersToBeUsed=[[markersToBeUsed, 'x'*ones(18,1)];[markersToBeUsed, 'y'*ones(18,1)];[markersToBeUsed, 'z'*ones(18,1)]];
markersToBeUsed=mat2cell(markersToBeUsed,ones(54,1),5);
eventsToBeUsed={'LHS','RTO','RHS','LTO'};

for sub=1:length(subs)
   %Load subject, initialize
   subs(sub)
   clear motionDataSub motionTimeSub metaDataSub gaitEventsSub gaitTimeSub
   load(['../data/OG' num2str(subs(sub)) 'forceEvents.mat'])
   i=0;
   %% Search for relevant trials & save them
   
   for j=1:length(procExpData.data)
       if ~isempty(procExpData.data{j}) &&  strcmp(procExpData.data{j}.metaData.type,'TM') && ~isempty(strfind(lower(procExpData.data{j}.metaData.name),'base'))
                i=i+1;
                markerData=procExpData.data{j}.markerData;
                % Get ref marker
                refMarker3D=.5*sum(markerData.getOrientedData({'LHIP','RHIP'}),2); %midHip

                %Ref axis option 1 (ideal): Body reference
                refAxis=squeeze(diff(markerData.getOrientedData({'LHIP','RHIP'}),1,2)); %L to R

                rotatedMarkerData=markerData.translate(-squeeze(refMarker3D)).alignRotate(refAxis,[0,0,1]);
                
               %Save trials
               motionDataSub{i} = rotatedMarkerData.getDataAsVector(markersToBeUsed);
               motionTimeSub{i}= rotatedMarkerData.Time;
               metaDataSub{i}=procExpData.data{j}.metaData;
               gaitEventsSub{i}= procExpData.data{j}.gaitEvents.getDataAsVector(eventsToBeUsed);
               gaitTimeSub{i}=procExpData.data{j}.gaitEvents.Time;
        
       end
   end
   
   %% Save the subject data
   motionData{sub}=motionDataSub;
   motionTime{sub}=motionTimeSub;
   metaData{sub}=metaDataSub;
   gaitEvents{sub}=gaitEventsSub;
   gaitTime{sub}=gaitTimeSub;
end
    
%% Save the cell array data
save ../data/allData_preProc.mat motionData motionTime metaData gaitEvents gaitTime markersToBeUsed eventsToBeUsed subs

%% Get the data into a super-array by taking the first 100k samples of each trial (100sec)
motionArray=nan(100001,54,length(subs),3);
eventArray=false(100001,4,length(subs),3);
Time=motionTime{1}{1}; %Assuming same time for everyone, which is probably true
for sub=1:length(subs)
    for j=1:3
        clear aux
        for k=1:size(motionData{sub}{j},2)
            aux(:,k)=interp1(motionTime{sub}{j},motionData{sub}{j}(:,k),gaitTime{sub}{j},'spline','extrap'); %Upsampling from motion data capture rate (100Hz) to gaitEvent rate (1000Hz)
        end
        %aux=bsxfun(@minus,aux,mean(aux,1)); %Centering
        %aux=bsxfun(@rdivide,aux,sqrt(prctile(aux.^2,95))); %Standarization
        motionArray(:,:,sub,j)=aux(1:100001,:);
        eventArray(:,:,sub,j)=gaitEvents{sub}{j}(1:100001,:);
    end
end

%% Save super-array
save ../data/dataArrays_upSampled_preProc.mat motionArray eventArray Time markersToBeUsed eventsToBeUsed subs
gi
%% Just for check: visualize marker data averaged across subjects & trials
figure
aa=reshape(mean(mean(motionArray,4),3),100001,18,3);
hold on
for i=1:18
    plot3(aa(:,i,1),aa(:,i,2),aa(:,i,3))
end