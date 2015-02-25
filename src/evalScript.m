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
output = circshift(roundedEventArray, 2);
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
% Calculate stats
%
metrics = zeros(2,nEventTypes,nSubjects,nTrials);
for i = 1:nEventTypes
    for j = 1:nSubjects
        for k = 1:nTrials
            y = l((m==i)&(n==j)&(o==k));
            yhat = p((q==i)&(r==j)&(s==k));
            assert(isequal(size(y),size(yhat)));
            % TODO what if not the same number of events?
            metrics(1,i,j,k) = norm(y-yhat,1); % L1 loss
            metrics(2,i,j,k) = norm(y-yhat,2); % L2 loss
            metrics(3,i,j,k) = sum(y~=yhat); % 0-1 loss
        end
    end
end

%
% Pad array
%
% padsize = 100;
% outputPadded = padarray(output,padsize);
% outputPadded(1,:,:,:) = 1;
% outputPadded(end,:,:,:) = 1;
% roundedPadded = padarray(roundedEventArray,padsize);
% roundedPadded(1,:,:,:) = 1;
% roundedPadded(end,:,:,:) = 1;

%% TODO window method
% hn = 0.5*ones(1,2);
% for i = 1:nEventTypes
%     for j = 1:nSubjects
%         for k = 1:nTrials
%             xn = l((m==i)&(n==j)&(o==k));
%             ixCycles = round(conv(xn,hn,'same'));
%             for p = 1:size(ixCycles)-2
%                 assert(sum(outputPadded(ixCycles(p):ixCycles(p+1),i,j,k))==1)
%             end
%         end
%     end
% end
