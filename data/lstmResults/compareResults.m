function compareResults(y,methodNames,varargin)
%varargin should be several matrices of same size as y, containing
%predictedY's

events=diff(double(y),[],1);
actualEvents(1,:)=sum(sum(events==-1,1),3); %TO
actualEvents(2,:)=sum(sum(events==1,1),3); %HS

win(:,1,1)=[.1:.1:1.9]; %triangular window looking  9 samples to each side of the center (10 samples out is equivalent to missing the event)
smoothEvents=convn(events,win,'same');
M=nargin-2;
for i=1:M
    predictedEvents{i}=diff(double(varargin{i}),[],1);
    eventCount(i,1,:)=sum(sum(predictedEvents{i}==-1,1),3); %TO detected by algorithm
    eventCount(i,2,:)=sum(sum(predictedEvents{i}==1,1),3); %HS detected by algorithm
    aux{i}=convn(predictedEvents{i},win,'same');
    for j=1:2 %TO and HS are treated sep
        err{i,j}=nan(size(events));
        err{i,j}(events==(2*j-3))=events(events==(2*j-3))-aux{i}(events==(2*j-3));
        %misses(i,j,:)=sum(sum(abs(err{i,j})>=1,1),3); %Count events that fall outside window directly (fn)
        misses(i,j,:)=sum(sum(events==(2*j-3) & sign(aux{i})~=(2*j-3),1),3); %Count events that fall outside window directly
        err{i,j}(abs(err{i,j})>=1)=nan; %So we don't include them in the error measure
        tp(i,j,:)=sum(sum(~isnan(err{i,j}),1),3); %Total number of actual events that have an error associated with them (meaning that the event was detected, it is just a matter of accuracy)       
    end
    if isempty(methodNames)
    methodNames{i}=['Method ' num2str(i)];
    end
end
falsep=eventCount-tp; %Number of false positives: events detected -tp




%% Plot results
figure
clear aa
subplot(4,1,1)
hold on
legStr={};
for i=1:M
    aa(i,:,:)=squeeze(err{i,1}(:,1,:));
    legStr{i}=[methodNames{i} ', \mu =' num2str(100*nanmean(aa(i,:)),2) ', \sigma=' num2str(100*nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['L leg TO (fn=' num2str(misses(:,1,1)') ', fp =' num2str(falsep(:,1,1)') ')'])
hold off
legend(legStr);

clear aa
subplot(4,1,3)
hold on
for i=1:M
    aa(i,:,:)=squeeze(err{i,2}(:,1,:));
    legStr{i}=[methodNames{i} ', \mu =' num2str(100*nanmean(aa(i,:)),2) ', \sigma=' num2str(100*nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['L leg HS (fn=' num2str(misses(:,2,1)') ', fp =' num2str(falsep(:,2,1)') ')'])
hold off
legend(legStr);

clear aa
subplot(4,1,2)
hold on
for i=1:M
    aa(i,:,:)=squeeze(err{i,1}(:,2,:));
    legStr{i}=[methodNames{i} ', \mu =' num2str(100*nanmean(aa(i,:)),2) ', \sigma=' num2str(100*nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['R leg TO (fn=' num2str(misses(:,1,2)') ', fp =' num2str(falsep(:,1,2)') ')'])
hold off
legend(legStr);

clear aa
subplot(4,1,4)
hold on
for i=1:M
    aa(i,:,:)=squeeze(err{i,2}(:,2,:));
    legStr{i}=[methodNames{i} ', \mu =' num2str(100*nanmean(aa(i,:)),2) ', \sigma=' num2str(100*nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['R leg HS (fn=' num2str(misses(:,2,2)') ', fp =' num2str(falsep(:,2,2)') ')'])
hold off
legend(legStr);
end

