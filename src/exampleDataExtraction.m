%% load some .mat file
load %Each file contains one variable called expData

%% Extract data for a single trial
trial=3; % Let's assume we want the data for trial #3
trialData=expData.data{trial}; %This gets all the data that was recorded for the given trial, it is an object of type labData

%% Extract both events and motion-capture data
eventData=trialData.gaitEvents; %This gets the event data that was detected, and is stored as a labTimeSeries object
motionData=trialData.markerData; %Also a labTimeSeries object

%% Now, get the time-vector and the actual data for each
eventTime=eventData.Time;
eventData=eventData.Data;

motionTime=motionData.Time;
motionData=motionData.Data;