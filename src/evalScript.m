%
% Eval script for ML project
%
clear
clf

% Load ground truth
pathToGround = '../../data/dataArrays.mat';
load(pathToGround);

% Load  output
% pathToOutput = '';
% load(pathToOutput);

% TODO this is a workaround to test evalScript
output= roundedEventArray;
output(:,1,:,:) = circshift(output(:,1,:,:), 0);
output(:,2,:,:) = circshift(output(:,2,:,:), 1);
output(:,3,:,:) = circshift(output(:,3,:,:), 2);
output(:,4,:,:) = circshift(output(:,4,:,:), 3);
output(find(output(:,1,1,1),1),1,1,1) = 0;

% Plot True Errors
trueErrors = computeTrueErrors(roundedEventArray, output);
[nEventTypes nSubjects nTrials] = size(trueErrors);

figure('visible','off')
% figure(1)
for i = 1:nEventTypes
    subplot(2,2,i)
    hist(cat(1, trueErrors{i,:,:}))
    title(['Event type #',int2str(i)])
    xlabel('True Error (frames)')
    ylabel('Frequency')
end
saveas(gcf, '../figures/trueErrorsPerEventType', 'png')

figure('visible','off')
% figure(2)
for i = 1:nEventTypes
    for j = 1:nSubjects
        subplot(nSubjects,nEventTypes,4*(i-1)+j)
        hist(cat(1, trueErrors{i,j,:}))
        % title(strcat(['Subject #',int2str(j)],['Event type #',int2str(i)]))
        title(['Subject #',int2str(j),' Event type #',int2str(i)])
        xlabel('True Error (frames)')
        ylabel('Frequency')
    end
end
saveas(gcf, '../figures/trueErrorsPerEventTypeAndSubject', 'png')
