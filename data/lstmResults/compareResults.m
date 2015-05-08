function compareResults(y,methodNames,varargin)
%varargin should be several matrices of same size as y, containing
%predictedY's

events=diff(double(y),[],1);



win(:,1,1)=[.1:.1:1.9]; %triangular window looking  9 samples to each side of the center (10 samples out is equivalent to missing the event)

M=nargin-2;
for i=1:M
    predictedEvents{i}=diff(double(varargin{i}),[],1);
    aux{i}=convn(predictedEvents{i},win,'same');
    for j=1:2 %TO and HS are treated sep
        err{i,j}=nan(size(events));
        err{i,j}(events==(2*j-3))=events(events==(2*j-3))-aux{i}(events==(2*j-3));
        misses(i,j,:)=sum(sum(abs(err{i,j})>=1,1),3); %Count events that fall outside window directly
        err{i,j}(abs(err{i})>=1)=nan; %So we don't include them in the error measure
        falsep{i,j}=nan; %Don't know how to quantify false positives easily
    end
    if isempty(methodNames)
    methodNames{i}=['Method ' num2str(i)];
    end
end


%% Plot results
figure
subplot(2,2,1)
hold on
legStr={};
for i=1:M
    aa(i,:,:)=squeeze(err{i,1}(:,1,:));
    legStr{i}=[methodNames{i} ', \mu =' num2str(nanmean(aa(i,:)),2) ', \sigma=' num2str(nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['L leg TO (misses=' num2str(misses(:,1,1)') ')'])
hold off
legend(legStr);

subplot(2,2,3)
hold on
for i=1:M
    aa(i,:,:)=squeeze(err{i,2}(:,1,:));
    legStr{i}=[methodNames{i}  ', \mu =' num2str(nanmean(aa(i,:)),2) ', \sigma=' num2str(nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['L leg HS (misses=' num2str(misses(:,2,1)') ')'])
hold off
legend(legStr);

subplot(2,2,2)
hold on
for i=1:M
    aa(i,:,:)=squeeze(err{i,1}(:,2,:));
    legStr{i}=[methodNames{i}  ', \mu =' num2str(nanmean(aa(i,:)),2) ', \sigma=' num2str(nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['R leg TO (misses=' num2str(misses(:,1,2)') ')'])
hold off
legend(legStr);

subplot(2,2,4)
hold on
for i=1:M
    aa(i,:,:)=squeeze(err{i,2}(:,2,:));
    legStr{i}=[methodNames{i}  ', \mu =' num2str(nanmean(aa(i,:)),2) ', \sigma=' num2str(nanstd(aa(i,:)),2)];
end
hist(aa(:,:)',[-.9:.1:.9])
set(gca,'XTick',[-.9:.1:.9],'XTickLabel',[-90:10:90])
xlabel('Event error in ms')
title(['R leg TO (misses=' num2str(misses(:,2,2)') ')'])
hold off
legend(legStr);
end

