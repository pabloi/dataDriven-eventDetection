%% load some .mat file
load ... %Fill this with the location of the .mat file
%Each file contains one variable called expData or procExpData
%Each of those files contains ALL the data recorded for a SINGLE subject.
%Within each of these there may be 

%% Extract data for a single trial
%Each trial contains a run of data capture (continuously recorded data).
%Within each trial, walking conditions

trial=5; % Let's assume we want the data for trial #5
trialData=expData.data{trial}; %This gets all the data that was recorded for the given trial, it is an object of type labData
if ~strcmp(trialData.metaData.type,'TM') %This field stores the information of whether a particular trial was in the treadmill (TM) or overground (OG)
    warning('This is not a treadmill (TM) trial, the events are not reliable!')
end

%% Extract both events and motion-capture data
eventData=trialData.gaitEvents; %This gets the event data that was detected, and is stored as a labTimeSeries object
motionData=trialData.markerData; %Also a labTimeSeries object

%Now, if you want, you can make a pretty plot (if this is too heavy, you can try using only some of the labels):
motionData2=motionData.split(10,20); %Getting data between t=10s and t=20s
eventData2=eventData.split(10,20);
labels=motionData.labels;
%labels=labels(1:6); %To plot only the first 6 columns of data, with
%correspond to the 3D coordinates of 2 markers
plot(motionData2,[],labels,[],eventData2)

%% Now, get the time-vector and the actual data for each so you can start doing stuff on it directly
eventTime=eventData.Time;
eventData=eventData.Data;

motionTime=motionData.Time;
motionData=motionData.Data;