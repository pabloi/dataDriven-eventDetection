%
% Eval script for ML project
%

%
% Parameters
%
fs = 100; % sampling frequency in Hertz
Ts = 1/fs; % sampling period in seconds
windowSizeSeconds = 100e-3; % window size in seconds
windowSizeFrames = windowSizeSeconds/Ts; % window size in number of frames

%
% Load ground truth
%
pathToGround = '../../data/dataArrays.mat';
load(pathToGround);
[nSamples nEventTypes nSubjects nTrials] = size(roundedEventArray);

%
% Load  output
%
% pathToOutput = '';
% load(pathToOutput);
% TODO this is a workaround to test evalScript
% output = circshift(roundedEventArray, 0);
output= roundedEventArray;
output(:,1,:,:) = circshift(output(:,1,:,:), 0);
output(:,2,:,:) = circshift(output(:,2,:,:), 1);
output(:,3,:,:) = circshift(output(:,3,:,:), 2);
output(:,4,:,:) = circshift(output(:,4,:,:), 3);
output(find(output(:,1,1,1),1),1,1,1) = 0;
if ~isequal(size(roundedEventArray), size(output))
    msgID = 'MATLAB:inputError';
    msg = 'Input size differs from ground truth.';
    baseException = MException(msgID,msg);
    throw(baseException)
end

%
% Get event indices
%
[l m n o] = ind2sub(size(roundedEventArray),find(roundedEventArray));
[p q r s] = ind2sub(size(output),find(output));

%
% Compute aggregate metrics: L1/L2/0-1 loss
%
loss = zeros(3,nEventTypes,nSubjects,nTrials);
for i = 1:nEventTypes
    for j = 1:nSubjects
        for k = 1:nTrials
            y = l((m==i)&(n==j)&(o==k));
            yhat = p((q==i)&(r==j)&(s==k));
            if isequal(size(y),size(yhat))
                loss(1,i,j,k) = norm(y-yhat,1)/length(y); % L1 loss
                loss(2,i,j,k) = norm(y-yhat,2)/length(y); % L2 loss
                loss(3,i,j,k) = sum(y~=yhat)/length(y); % 0-1 loss
            else
                loss(1,i,j,k) = NaN;
                loss(2,i,j,k) = NaN;
                loss(3,i,j,k) = NaN;
            end
        end
    end
end

display('Average L1/L2/0-1 Loss for LHS RTO RHS LTO')
display(mean(mean(loss,4),3))

display('Per subject/trial L1/L2/0-1 Loss for LHS RTO RHS LTO')
display(loss)

%
% Compute per cycle metrics: tp fp tn fn
%
maxNEvents = 100;
hn = 0.5*ones(1,2);
metrics = zeros(4,nEventTypes,nSubjects,nTrials);
errors = NaN(maxNEvents,nEventTypes,nSubjects,nTrials);
for i = 1:nEventTypes
    for j = 1:nSubjects
        for k = 1:nTrials
            y = l((m==i)&(n==j)&(o==k));
            yAugmented = [1 y' nSamples]';
            yhat = p((q==i)&(r==j)&(s==k));
            % yhatAugmented = padarray(yhat,length(y)-length(yhat));
            yCycles = round(conv(yAugmented,hn,'valid'));
            for t = 1:length(yCycles)-1
                nbhat = sum(output(yCycles(t):yCycles(t+1),i,j,k));
                if nbhat > 0
                    metrics(1,i,j,k) = metrics(1,i,j,k) + 1;
                    metrics(2,i,j,k) = metrics(2,i,j,k) + (nbhat - 1);
                    metrics(3,i,j,k) = metrics(3,i,j,k) + 0;
                else
                    metrics(1,i,j,k) = metrics(1,i,j,k) + 0;
                    metrics(2,i,j,k) = metrics(2,i,j,k) + 0;
                    metrics(3,i,j,k) = metrics(3,i,j,k) + 1;
                end

                if (nbhat == 1) & (t <= length(yhat))
                    errors(t,i,j,k) = y(t) - yhat(t);
                else
                    errors(t,i,j,k) = NaN;
                end
            end
        end
    end
end

% display('TP FP FN for LHS RTO RHS LTO')
% display(mean(mean(metrics,4),3))

display('Per subject/trial TP FP FN for LHS RTO RHS LTO')
display(metrics)

% TODO plot histogram of TE
