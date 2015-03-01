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
% output(find(output(:,1,1,1),1),1,1,1) = 0;
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
% Aggregate Metrics: L1/L2/0-1 Loss
%
metrics = zeros(2,nEventTypes,nSubjects,nTrials);
for i = 1:nEventTypes
    for j = 1:nSubjects
        for k = 1:nTrials
            y = l((m==i)&(n==j)&(o==k));
            yhat = p((q==i)&(r==j)&(s==k));
            % assert(isequal(size(y),size(yhat)));
            if isequal(size(y),size(yhat))
                metrics(1,i,j,k) = norm(y-yhat,1)/size(y,1); % L1 loss
                metrics(2,i,j,k) = norm(y-yhat,2)/size(y,1); % L2 loss
                metrics(3,i,j,k) = sum(y~=yhat)/size(y,1); % 0-1 loss
            else
                metrics(1,i,j,k) = NaN;
                metrics(2,i,j,k) = NaN;
                metrics(3,i,j,k) = NaN;
            end
        end
    end
end

display('Aggregate L1/L2/0-1 Loss for LHS RTO RHS LTO')
display(mean(mean(metrics,4),3))

display('Per subject/trial L1/L2/0-1 Loss for LHS RTO RHS LTO')
display(metrics)

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
