%
% Eval script for ML project
%
clear

% Plotting
plotting = false; % plot figures yes/no
saveplot = true; % save figures to file yes/no
figpath = '../../figures/';

% Load ground truth
% pathToGround = '../../data/dataArrays_upSampled.mat';
pathToGround = '../../data/dataArrays.mat';
load(pathToGround);
% gold = eventArray;
gold = roundedEventArray;

% Load  output
% pathToOutput = '../../data/fva_20150309_0956.mat';
pathToOutput = '../../data/fva_rounded_20150309_1259.mat';
load(pathToOutput);
% output = eventArrayFVA;
output = roundedEventArrayFVA;

% ------------------------------------------------------------------------------

% Define window size for false positives/negatives
windowSize = 21;
halfWindow = floor(windowSize/2);

% Compute True Errors
% trueErrors = computeTrueErrorsMask(gold, output); % compute TE using xcor
[trueErrors, metrics] = computeTrueErrorsRobust(gold, output, windowSize);
[nEventTypes nSubjects nTrials] = size(trueErrors);

% ------------------------------------------------------------------------------

% Histogram of true errors per event type
if plotting
    figure(1)
else
    figure('visible','off')
end
for i = 1:nEventTypes
    subplot(2,2,i)
    TE = cat(1, trueErrors{i,:,:});
    maxval = max(abs(TE));
    % hist(TE,[-(maxval+1):maxval+1]);
    hist(TE,[-halfWindow:halfWindow]);
    title([
        'Event Type ',...
        int2str(i),...
        ' (window = ',...
        int2str(windowSize),...
        ' frames)'...
        ])
    xlabel([...
        'True Error (frames) (n=',...
        int2str(length(TE)),...
        ', na=',...
        ')'...
        ])
    ylabel('Frequency')
end
if saveplot
    saveas(gcf, strcat(figpath,'trueErrorsPerEventType'), 'png')
end

% Histogram of true errors per event type and subject
if plotting
    figure(2)
else
    figure('visible','off')
end
for i = 1:nEventTypes
    for j = 1:nSubjects
        % subplot(nSubjects,nEventTypes,nSubjects*(i-1)+j)
        subplot(nSubjects,nEventTypes,nEventTypes*(j-1)+i)
        TE = cat(1, trueErrors{i,j,:});
        maxval = max(abs(TE));
        % hist(TE,[-(maxval+1):maxval+1]);
        hist(TE,[-halfWindow:halfWindow]);
        title([
            'Subject ',...
            int2str(j),...
            ' Event Type ',...
            int2str(i)...
            ' (window = ',...
            int2str(windowSize),...
            ' frames)'...
            ])
        xlabel([...
            'True Error (frames) (n=',...
            int2str(length(TE)),...
            ', na=',...
            ')'...
            ])
        ylabel('Frequency')
    end
end
if saveplot
    saveas(gcf, strcat(figpath,'trueErrorsPerEventTypeAndSubject'), 'png')
end

% ------------------------------------------------------------------------------

% Histogram of absolute errors per event type
if plotting
    figure(3)
else
    figure('visible','off')
end
for i = 1:nEventTypes
    subplot(2,2,i)
    AE = abs(cat(1, trueErrors{i,:,:}));
    maxval = max(AE);
    % hist(AE,[0:maxval+1]);
    hist(AE,[0:halfWindow]);
    title([
        'Event Type ',...
        int2str(i),...
        ' (window = ',...
        int2str(windowSize),...
        ' frames)'...
        ])
    xlabel([...
        'Absolute Error (frames) (n=',...
        int2str(length(AE)),...
        ', na=',...
        ')'...
        ])
    ylabel('Frequency')
end
if saveplot
    saveas(gcf, strcat(figpath,'absoluteErrorsPerEventType'), 'png')
end

% Histogram absolute errors per event type and subject
if plotting
    figure(4)
else
    figure('visible','off')
end
for i = 1:nEventTypes
    for j = 1:nSubjects
        % subplot(nSubjects,nEventTypes,nSubjects*(i-1)+j)
        subplot(nSubjects,nEventTypes,nEventTypes*(j-1)+i)
        AE = abs(cat(1, trueErrors{i,j,:}));
        maxval = max(AE);
        % hist(AE,[0:maxval+1]);
        hist(AE,[0:halfWindow]);
        title([
            'Subject ',...
            int2str(j),...
            ' Event Type ',...
            int2str(i),...
            ' (window = ',...
            int2str(windowSize),...
            ' frames)'...
            ])
        xlabel([...
            'Absolute Error (frames) (n=',...
            int2str(length(AE)),...
            ', na=',...
            ')'...
            ])
        ylabel('Frequency')
    end
end
if saveplot
    saveas(gcf, strcat(figpath,'absoluteErrorsPerEventTypeAndSubject'), 'png')
end

% ------------------------------------------------------------------------------

% Boxplot of true errors per event type
if plotting
    figure(5)
else
    figure('visible','off')
end
TE = cat(1, trueErrors{:,:,:});
groupsize = [];
for i = 1:nEventTypes
    groupsize = [groupsize i*ones(1,length(cat(1, trueErrors{i,:,:})))];
end
boxplot(TE,groupsize)
title([
    'True Error per Event Type',...
    ' (window = ',...
    int2str(windowSize),...
    ' frames)'...
    ])
xlabel('Event Type')
ylabel('True Error (frames)')
if saveplot
    saveas(gcf, strcat(figpath,'trueErrorsBoxplot'), 'png')
end

% Boxplot of true errors per event and subject
if plotting
    figure(6)
else
    figure('visible','off')
end
for i = 1:nEventTypes
    subplot(nEventTypes,1,i)
    TE = cat(1, trueErrors{i,:,:});
    groupsize = [];
    for k = 1:nSubjects
        groupsize = [groupsize k*ones(1,length(cat(1, trueErrors{i,k,:})))];
    end
    boxplot(TE,groupsize)
    title([
        'Event Type ',...
        int2str(i),...
        ' (window = ',...
        int2str(windowSize),...
        ' frames)'...
        ])
    xlabel('Subject')
    ylabel('True Error (frames)')
end
if saveplot
    saveas(gcf, strcat(figpath,'trueErrorsPerEventTypeBoxplot'), 'png')
end

% Boxplot of true errors per subject
if plotting
    figure(7)
else
    figure('visible','off')
end
for i = 1:nSubjects
    subplot(nSubjects,1,i)
    TE = cat(1, trueErrors{:,i,:});
    groupsize = [];
    for k = 1:nEventTypes
        groupsize = [groupsize k*ones(1,length(cat(1, trueErrors{k,i,:})))];
    end
    boxplot(TE,groupsize)
    title([
        'Subject ',...
        int2str(i),...
        ' (window = ',...
        int2str(windowSize),...
        ' frames)'...
        ])
    xlabel('Event Type')
    ylabel('True Error (frames)')
end
if saveplot
    saveas(gcf, strcat(figpath,'trueErrorsPerSubjectBoxplot'), 'png')
end

% Boxplot true errors per event type and subject
if plotting
    figure(8)
else
    figure('visible','off')
end
for i = 1:nEventTypes
    for j = 1:nSubjects
        % subplot(nSubjects,nEventTypes,nSubjects*(i-1)+j)
        subplot(nSubjects,nEventTypes,nEventTypes*(j-1)+i)
        TE = cat(1, trueErrors{i,j,:});
        boxplot(TE)
        title([
            'Subject ',...
            int2str(j),...
            ' Event Type ',....
            int2str(i),...
            ' (window = ',...
            int2str(windowSize),...
            ' frames)'...
            ])
        ylabel('TE (frames)')
    end
end
if saveplot
    saveas(gcf, strcat(figpath,'trueErrorsPerEventTypeAndSubjectBoxplot'), ...
    'png')
end

% ------------------------------------------------------------------------------
