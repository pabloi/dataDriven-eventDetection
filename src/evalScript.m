%
% Eval script for ML project
%
clear

% Plotting
plotting = false; % plot figures yes/no
saveplot = false; % save figures to file yes/no
% figpath = '../../figures_upsampled/';
figpath = '../../figures_rounded/';
% figpath = '../../figures_stroke/';

% Results
results = true; % save results to file yes/no
% respath = '../../results_upsampled/';
respath = '../../results_rounded/';
% respath = '../../results_stroke/';

% Load ground truth
% pathToGround = '../../data/dataArrays_upSampled.mat';
pathToGround = '../../data/dataArrays.mat';
% pathToGround = '../../data/dataArrays_stroke.mat';
load(pathToGround);
% gold = eventArray;
gold = roundedEventArray;
% gold = eventArray(:,:,1:8,:);

% Load  output
% pathToOutput = '../../data/fva_20150309_0956.mat';
pathToOutput = '../../data/fva_rounded_20150309_1259.mat';
% pathToOutput = '../../data/fva_stroke_20150313_0850.mat';
load(pathToOutput);
% output = eventArrayFVA;
output = roundedEventArrayFVA;
% output = eventArrayFVA(:,:,1:8,:);

% ------------------------------------------------------------------------------

% Define window size for false positives/negatives
windowSize = 21;
% windowSize = 201;
halfWindow = floor(windowSize/2);

% Compute True Errors
% trueErrors = computeTrueErrors(gold, output); % compute TE
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
    xlim([-halfWindow-1 halfWindow+1]);
    title([
        'ET ',...
        int2str(i),...
        ' (window = ',...
        int2str(windowSize),...
        ' frames)'...
        ])
    xlabel([...
        'TE (frames) (n=',...
        int2str(length(TE)),...
        ', tp=',...
        int2str(sum(sum(metrics(1,i,:,:)))),...
        ', fn=',...
        int2str(sum(sum(metrics(3,i,:,:)))),...
        ', fp=',...
        int2str(sum(sum(metrics(2,i,:,:)))),...
        ', other=',...
        int2str(sum(sum(metrics(4,i,:,:)))),...
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
        xlim([-halfWindow-1 halfWindow+1]);
        title([
            'Subject ',...
            int2str(j),...
            ' ET ',...
            int2str(i)...
            ' (window = ',...
            int2str(windowSize),...
            ' frames)'...
            ])
        xlabel([...
            'TE (fr) (n=',...
            int2str(length(TE)),...
            ', tp=',...
            int2str(sum(metrics(1,i,j,:))),...
            ', fn=',...
            int2str(sum(metrics(3,i,j,:))),...
            ', fp=',...
            int2str(sum(metrics(2,i,j,:))),...
            ', ot=',...
            int2str(sum(metrics(4,i,j,:))),...
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
        'ET ',...
        int2str(i),...
        ' (window = ',...
        int2str(windowSize),...
        ' frames)'...
        ])
    xlabel([...
        'AE (frames) (n=',...
        int2str(length(AE)),...
        ', tp=',...
        int2str(sum(sum(metrics(1,i,:,:)))),...
        ', fn=',...
        int2str(sum(sum(metrics(3,i,:,:)))),...
        ', fp=',...
        int2str(sum(sum(metrics(2,i,:,:)))),...
        ', other=',...
        int2str(sum(sum(metrics(4,i,:,:)))),...
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
            ' ET ',...
            int2str(i),...
            ' (window = ',...
            int2str(windowSize),...
            ' frames)'...
            ])
        xlabel([...
            % 'AE (frames) (n=',...
            'AE (fr) (n=',...
            int2str(length(AE)),...
            ', tp=',...
            int2str(sum(metrics(1,i,j,:))),...
            ', fn=',...
            int2str(sum(metrics(3,i,j,:))),...
            ', fp=',...
            int2str(sum(metrics(2,i,j,:))),...
            ', or=',...
            int2str(sum(metrics(4,i,j,:))),...
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
    'TE per Event Type',...
    ' (window = ',...
    int2str(windowSize),...
    ' frames)'...
    ])
xlabel('Event Type')
ylabel('TE (frames)')
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
    ylabel('TE (frames)')
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
    ylabel('TE (frames)')
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

if results
    dlm = '\t';
    fileID = fopen(strcat(respath,'/medianMeanStdPerEventType.csv'),'w');
    header = ['event type',dlm,'median',dlm,'mean',dlm,'std','\n'];
    fprintf(fileID,header);
    for i = 1:nEventTypes
        TE = cat(1,trueErrors{i,:,:});
        stats = [median(TE) mean(TE) std(TE)];
        fprintf(fileID, '%d\t%f\t%f\t%f\n', i, stats);
    end
    fclose(fileID);

    fileID = fopen(strcat(respath,'/medianMeanStdPerSubject.csv'),'w');
    header = ['subject',dlm,'median',dlm,'mean',dlm,'std','\n'];
    fprintf(fileID,header);
    for i = 1:nSubjects
        TE = cat(1,trueErrors{:,i,:});
        stats = [median(TE) mean(TE) std(TE)];
        fprintf(fileID, '%d\t%f\t%f\t%f\n', i, stats);
    end
    fclose(fileID);

    fileID = fopen(strcat(respath,'/medianMeanStdPerSubjectAndEventType.csv'),'w');
    header = ['subject',dlm,'event type',dlm,'median',dlm,'mean',dlm,'std','\n'];
    fprintf(fileID,header);
    for i = 1:nSubjects
        for j = 1:nEventTypes
            TE = cat(1,trueErrors{j,i,:});
            stats = [median(TE) mean(TE) std(TE)];
            fprintf(fileID, '%d\t%d\t%f\t%f\t%f\n', i, j, stats);
        end
    end
    fclose(fileID);
end
