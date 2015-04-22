diary('./makeDataMatrixMaster_Arrays.log')

%%
clearvars
load ../data/allData_master.mat
%% Get the data into a super-array by taking the first 50secs of data @100Hz (5k samples)
N=5000;
motionArray=nan(N,54,length(motionData));
motionArrayUnproc=nan(N,54,length(motionData));
motionArrayStand=nan(N,54,length(motionData));
eventArray=false(N,4,length(motionData));
stanceArray=false(N,2,length(motionData));
names=cell(size(motionData));
Time=motionTime{1}; %Assuming same time for everyone, which is probably true
k=0;
for counter=1:length(motionData)
        k=k+1;
        [stanceL,stanceR] = getStanceFromEvents(gaitEvents{counter},eventsToBeUsed);
        T=length(motionTime{counter});
        %downsampledStance=nan(N,2);
        relIdxs=find(motionTime{counter}(1)==gaitTime{counter})+round(diff(motionTime{counter}(1:2))/diff(gaitTime{counter}(1:2)))*[0:T-1]; %Getting samples of stance data that correspond to first N samples of motionData
        %Check:
        if any(motionTime{counter} ~= gaitTime{counter}(relIdxs))
            error('Time vectors do not match.')
        end
        
        downSampledStance=full([stanceL(relIdxs),stanceR(relIdxs)]);
        [LHS,RHS,LTO,RTO] = getEventsFromStance(downSampledStance(:,1),downSampledStance(:,2));
        downSampledEvents= full([LHS,RTO,RHS,LTO]); %Same  order as eventsToBeUsed
        
        %Body-referenced data:
        aa=1:size(motionData{counter},1);
%         figure
%         hold on
%         for j=1:18
%         plot3(motionData{counter}(:,j),motionData{counter}(:,j+18),motionData{counter}(:,j+36),'b.');
%         end
        for j=1:54
            notNanIdxs=~isnan(motionData{counter}(:,j));
            if ~all(notNanIdxs)
                motionData{counter}(~notNanIdxs,j)=interp1(aa(notNanIdxs),motionData{counter}(notNanIdxs,j),aa(~notNanIdxs),'linear',NaN); %Interpolating NaNs
            end
        end
%         for j=1:18
%             plot3(motionData{counter}(:,j),motionData{counter}(:,j+18),motionData{counter}(:,j+36),'k');
%         end
%         hold off
            
        stInd=find(all(~isnan(motionData{counter}),2),1,'first');
        motionArray(:,:,k)=motionData{counter}(stInd:N+stInd-1,:);
        
        %Lab-referenced data:
        for j=1:54
            notNanIdxs=~isnan(motionDataUnrotated{counter}(:,j));
            if ~all(notNanIdxs)
                motionDataUnrotated{counter}(~notNanIdxs,j)=interp1(aa(notNanIdxs),motionDataUnrotated{counter}(notNanIdxs,j),aa(~notNanIdxs),'linear',NaN); %Interpolating NaNs
            end
        end
        motionArrayUnproc(:,:,k)=motionDataUnrotated{counter}(stInd:N+stInd-1,:);
        
        %Standardized samples:
        aux=bsxfun(@minus,motionArray(:,:,k),mean(motionArray(:,:,k),1));
        motionArrayStand(:,:,k)=bsxfun(@rdivide,aux,sqrt(prctile(aux.^2,95,1))+.0001); %Dividing by 95% percentile +.1mm for regularization
        
        %Events & stances:
        eventArray(:,:,k)=downSampledEvents(stInd:N+stInd-1,:,:);
        stanceArray(:,:,k)=(downSampledStance(stInd:N+stInd-1,:,:)==1);
        names{k}=[metaDatas{counter}.sub,'_',metaDatas{counter}.trialName];
        
        %Check for NaNs and discard if necessary
        if any(any(isnan(motionArrayUnproc(:,:,k))))
            warning(['There are trailing NaN samples. No 50secs of data are good. Discarding trial ' num2str(counter)])
            k=k-1;
        end
end

%% Save super-array
save ../data/dataArrays_master.mat motionArray* eventArray stanceArray Time markersToBeUsed eventsToBeUsed subs names

%% Just for check: visualize marker data averaged across subjects & trials
% figure('Name','Rotated data')
% aux=motionArray;
% % aux(motionArrayUnproc==0)=nan;
% % for i=1:size(motionArray,3)
% %  auxId=any(motionArrayUnproc(:,:,i)==0,2);
% %  aux(auxId,:,i)=nan;
% % end
% %aux(abs([diff(motionArrayUnproc);ones(1,size(motionArray,2),size(motionArray,3))])>20)=nan;
% hold on
% for i=1:size(motionArray,3)
% aa=reshape(aux(:,:,i),N,18,3);
% idx1=stanceArray(:,1,i)==1 & stanceArray(:,2,i)==0;
% idx2=stanceArray(:,1,i)==1 & stanceArray(:,2,i)==1;
% idx3=stanceArray(:,1,i)==0 & stanceArray(:,2,i)==1;
% idx4=stanceArray(:,1,i)==0 & stanceArray(:,2,i)==0;
% 
%     for j=1:18
%         
%         plot3(aa(idx1,j,1),aa(idx1,j,2),aa(idx1,j,3),'b')
%         plot3(aa(idx2,j,1),aa(idx2,j,2),aa(idx2,j,3),'r')
%         plot3(aa(idx3,j,1),aa(idx3,j,2),aa(idx3,j,3),'k')
%         plot3(aa(idx4,j,1),aa(idx4,j,2),aa(idx4,j,3),'g')
%     end
% end
% hold off
% 
% %% Just for check: visualize marker data averaged across subjects & trials
% figure('Name','Rotated & stand data')
% aux=motionArrayStand;
% % aux(motionArrayUnproc==0)=nan;
% % for i=1:size(motionArray,3)
% %  auxId=any(motionArrayUnproc(:,:,i)==0,2);
% %  aux(auxId,:,i)=nan;
% % end
% 
% hold on
% for i=1:size(motionArray,3)
% aa=reshape(aux(:,:,i),N,18,3);
% 
% for j=1:18
%     plot3(aa(:,j,1),aa(:,j,2),aa(:,j,3))
% end
% end
% hold off
% 
% %%
% figure('Name','Original data')
% aux=motionArrayUnproc;
% aux(motionArrayUnproc==0)=nan;
% hold on
% for i=1:size(motionArray,3)
% aa=reshape(aux(:,:,i),N,18,3);
% 
% for j=1:18
%     plot3(aa(:,j,1),aa(:,j,2),aa(:,j,3))
% end
% end
% hold off

%%
diary off