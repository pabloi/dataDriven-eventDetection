function [ trueErrors ] = computeTrueErrors( groundTruth, estimatedEvents )
    % groundTruth: nSamples*nEventTypes*nSubjects*nTrials array
    % estimatedEvents: nSamples*nEventTypes*nSubjects*nTrials array
    % trueErrors: nEventTypes*nSubjects*nTrials cell array

    % Check size
    if ~isequal(size(groundTruth), size(estimatedEvents))
        msgID = 'MATLAB:inputError';
        msg = 'Input size differs from ground truth.';
        baseException = MException(msgID,msg);
        throw(baseException)
    end

    % Get dimension
    [nSamples nEventTypes nSubjects nTrials] = size(groundTruth);

    % Get event indices
    [l m n o] = ind2sub(size(groundTruth),find(groundTruth));
    [p q r s] = ind2sub(size(estimatedEvents),find(estimatedEvents));

    % Compute true errors
    trueErrors = cell(nEventTypes,nSubjects,nTrials);
    for i = 1:nEventTypes
        for j = 1:nSubjects
            for k = 1:nTrials
                y = l((m==i)&(n==j)&(o==k));
                yhat = p((q==i)&(r==j)&(s==k));
                [acor,lag] = xcorr(yhat,y);
                [~,I] = max(abs(acor));
                timeDiff = lag(I);
                li = 1 + timeDiff;
                ui = li + length(y) - 1;
                trueErrors(i,j,k) = {y - yhat(li:ui)};
            end
        end
    end
end
