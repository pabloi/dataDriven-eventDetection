function [ trueErrors metrics ] = ...
    computeTrueErrorsRobus( groundTruth, estimatedEvents, windowSize )
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
    metrics = zeros(4,nEventTypes,nSubjects,nTrials);
    halfWindow = floor(windowSize/2);
    for i = 1:nEventTypes
        for j = 1:nSubjects
            for k = 1:nTrials
                y = l((m==i)&(n==j)&(o==k));
                yhat = estimatedEvents(:,i,j,k);
                nEvents = length(y);
                nEstimatedEvents = sum(yhat);
                errors = [];
                for h = 1:nEvents
                    li = max(1,y(h)-halfWindow);
                    lu = min(length(yhat),y(h)+halfWindow);
                    yhatWindow = yhat(li:lu);
                    nbhat = sum(yhatWindow);
                    if nbhat == 1
                        metrics(1,i,j,k) = metrics(1,i,j,k) + 1;
                        e = find(yhatWindow) - (halfWindow + 1);
                    elseif nbhat == 0
                        metrics(3,i,j,k) = metrics(3,i,j,k) + 1;
                    else
                        metrics(1,i,j,k) = metrics(1,i,j,k) + 1;
                        metrics(2,i,j,k) = metrics(2,i,j,k) + (nbhat - 1);
                        e = min(find(yhatWindow) - (halfWindow + 1));
                    end
                    errors = [errors; e];
                end
                metrics(4,i,j,k) = ...
                    nEstimatedEvents - nEvents - sum(metrics(2,i,j,k));
                trueErrors(i,j,k) = {errors};
            end
        end
    end
end
