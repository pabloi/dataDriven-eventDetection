subs={'20','21','23','35','36','37','40','41','0003','0008','P0001','P0002','P0003','P0004','P0005','P0006','P0007','P0008','P0009','P0010','P0011','P0012','P0013','P0014','P0015','P0016'}; %Ignoring subject 34 because for some reason the TM base trial is not found
diary('./makeDataMatrixMaster_loading.log')
%Initialize
clear motionData motionTime metaData gaitEvents gaitTime
markersToBeUsed={'TOE';'ANK';'HEE';'KNE';'KNEE';'HIP';'TIB';'SHANK';'THI';'PSI';'PSIS';'ASI';'ASIS'};
markersToBeUsed=[strcat('R',markersToBeUsed);strcat('L',markersToBeUsed)];
markersToBeUsed=[strcat(markersToBeUsed,'x');strcat(markersToBeUsed,'y');strcat(markersToBeUsed,'z')];
eventsToBeUsed={'LHS','RTO','RHS','LTO'};
metaDatas={};
counter=0;
for sub=1:length(subs)
   %Load subject, initialize
   subs(sub)
   clear procExpData
   try
        a=load(['../data/OG' subs{sub} 'forceEvents.mat']);
   catch
       a=load(['../../../../lab/synergies/matData/' subs{sub} '.mat']);
   end
   b=fields(a);
   procExpData=a.(b{1});
   %% Search for relevant trials & save them   
   for j=1:length(procExpData.data)
       if ~isempty(procExpData.data{j}) &&  strcmp(procExpData.data{j}.metaData.type,'TM')...
               && isempty(strfind(lower(procExpData.data{j}.metaData.name),'hill')) && (...
                ~isempty(strfind(lower(procExpData.data{j}.metaData.name),'base'))...
               || ~isempty(strfind(lower(procExpData.data{j}.metaData.name),'slow'))...
               || ~isempty(strfind(lower(procExpData.data{j}.metaData.name),'mid'))...
               || ~isempty(strfind(lower(procExpData.data{j}.metaData.name),'fast')))
           disp(['Trial ' num2str(j) ' (' procExpData.data{j}.metaData.name ') was selected.'])
           counter=counter+1;
           markerData=procExpData.data{j}.markerData;
           markerData.Data(markerData.Data==0)=nan; %Setting to NaNs all samples identical to 0 (this should never happen).
           % Get ref marker
           refMarker3D=.5*sum(markerData.getOrientedData({'LHIP','RHIP'}),2); %midHip
           %Ref axis option 1 (ideal): Body reference
           refAxis=squeeze(diff(markerData.getOrientedData({'LHIP','RHIP'}),1,2)); %L to R
           rotatedMarkerData=markerData.translate(-squeeze(refMarker3D)).alignRotate(refAxis,[0,0,1]);

           %Save data
           [motionDataUnrotated{counter},motionTime{counter},actualMarkers{counter}]=markerData.getDataAsVector(markersToBeUsed);
           motionData{counter} = rotatedMarkerData.getDataAsVector(markersToBeUsed);
           metaData{counter}=procExpData.data{j}.metaData;
           gaitEvents{counter}= procExpData.data{j}.gaitEvents.getDataAsVector(eventsToBeUsed);
           gaitTime{counter}=procExpData.data{j}.gaitEvents.Time;        
           metaDatas{counter}.sub=subs{sub};
           metaDatas{counter}.trial=j;
           metaDatas{counter}.trialName=procExpData.data{j}.metaData.name;
           if length(actualMarkers{counter})~=54
               warning('Not all markers are present. Skipping trial.')
               counter=counter-1;
           end
           
       end
   end
end
    
%% Save the cell array data
save ../data/allData_master.mat motionData* motionTime metaData gaitEvents gaitTime markersToBeUsed eventsToBeUsed subs actualMarkers metaDatas

%%
diary off